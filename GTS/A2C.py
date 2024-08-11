# Advantage Actor-Critic implmentation,
#
# heavily influenced by https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial
from typing import Any, Dict, List, Optional, Tuple, cast
import dataclasses
import datetime
import math
import numpy as np
import gymnasium as gym
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from enum import Enum

from .logger import Logger
from .params import ParamDict, bakeParams
from .parametricNet import ParametricNet, HiddenLayers, asTensor
from .stochasticNet import StochasticNetLearnedSD, StochasticNetFixedSD, StochasticNet
from .physicalModel import PhysicalModelSpec
from .physicalModels import fetchPhysicalModelSpec
from .score import computeScore
from .tester import test
from .trajectory import Trajectory

__all__ = ["A2C"]
"""The params used by A2C and their defaults."""
_paramSpecs: ParamDict = {
    "nEnvironments": 8,
    "actor.varianceScale": 0.08,
    "actor.varianceScaleDelta": 0.0,
    "actor.learningRate": 0.001,
    "actor.nHidden": 2,
    "actor.hiddenSize": 32,
    "actor.hiddenActivation": "ReLU",
    "critic.learningRate": 0.005,
    "critic.nHidden": 2,
    "critic.hiddenSize": 32,
    "critic.hiddenActivation": "ReLU",
    "lambda": 0.97,
    "discountFactor": 0.999,
    "maxEpisodeLength": 500,
    "entropyCoeff": 0.01,
    "seed": 1
}


################################################################################
class A2C:
    """Advantage Actor Critic learning algorithm"""
    actorNet: StochasticNetFixedSD
    criticNet: ParametricNet

    def __init__(self, physicalModelSpec: PhysicalModelSpec,
                 params: ParamDict) -> None:
        """A2C algorithm to train a mujoco simulation
        """
        self._abort = False
        self._params = bakeParams(params, _paramSpecs)

        self.physicalModelSpec = physicalModelSpec

        self.nEnvironments = cast(int, self._params["nEnvironments"])

        rng = np.random.default_rng(seed=cast(int, self._params["seed"]))

        # make the actor network
        hiddenLayers = HiddenLayers.fromDict(self._params, "actor")

        # get fixed variances for the stochastic network, based on the
        # range of the actions, add some noise onto that as well
        fixedVarianceFactor = cast(float, self._params["actor.varianceScale"])
        variances: List[List[float]] = []
        vsd = cast(float, self._params["actor.varianceScaleDelta"])

        varianceScaleMin = fixedVarianceFactor * (1 - vsd)
        varianceScaleDelta = vsd * 2
        # make a set of slightly randomised variances per enviroment
        for i in range(self.nEnvironments):
            theseVariances: List[float] = []
            for actionRange in physicalModelSpec.actionRanges():
                actionDelta = actionRange[1] - actionRange[0]
                rando = 2.0 * rng.random() - 1.0
                thisScale = varianceScaleMin + varianceScaleDelta * rng.random(
                )
                theseVariances.append(actionDelta * thisScale)
            variances.append(theseVariances)

        self.actorNet = StochasticNetFixedSD(physicalModelSpec.nObservations(),
                                             physicalModelSpec.nActions(),
                                             hiddenLayers, variances)
        lr: float = cast(float, self._params["actor.learningRate"])
        self.actorOptimiser = torch.optim.AdamW(self.actorNet.parameters(),
                                                lr=lr)

        # entopy is fixed as variances are fixed
        entropy = 0.5 * np.log(2 * math.pi * np.array(variances)**2) + 0.5
        self.entropy = torch.tensor(entropy)

        # critic
        hiddenLayers = HiddenLayers.fromDict(self._params, "critic")
        self.criticNet = ParametricNet(physicalModelSpec.nObservations(), 1,
                                       hiddenLayers)
        lr = cast(float, self._params["critic.learningRate"])
        self.criticOptimiser = torch.optim.AdamW(self.criticNet.parameters(),
                                                 lr=lr)
        self._score: Dict[str, float] = {}

    def paramSpecs(self) -> ParamDict:
        return _paramSpecs

    def modelSpec(self) -> PhysicalModelSpec:
        return self.physicalModelSpec

    def policyNetwork(self) -> StochasticNet:
        return self.actorNet

    def score(self) -> Dict[str, float]:
        return self._score

    def setScore(self, score: Dict[str, float]) -> None:
        self._score = score

    def abort(self) -> None:
        self._abort = True

    def aborted(self) -> bool:
        return self._abort

    def params(self) -> Dict[str, Any]:
        return self._params

    def agentName(self) -> str:
        """Name of this trainer."""
        return "A2C"

    def asDict(self) -> Dict[str, Any]:
        """Retrieve the dict needed for serialisation."""
        values: Dict[str, Any] = {
            "trainer": "A2C",
            "model": self.physicalModelSpec.id(),
            "params": self._params,
            "policy.net": self.actorNet.asDict(),
            "policy.optimiserState": self.actorOptimiser.state_dict(),
            "critic.net": self.criticNet.asDict(),
            "critic.optimiserState": self.criticOptimiser.state_dict(),
            "score": self._score,
        }
        return values

    def save(self, path: str) -> None:
        """Save this parameterised network out to the given file."""
        torch.save(self.asDict(), path)

    @staticmethod
    def loadFromDict(values: Dict[str, Any]) -> "A2C":
        """Load state from a serialisation dict"""
        assert values["trainer"] == "A2C"
        physicalModel = fetchPhysicalModelSpec(values["model"])
        params = values["params"]
        reinforce = A2C(physicalModel, params)
        reinforce.actorNet.load_state_dict(values["policy.net"]["state"])
        reinforce.actorOptimiser.load_state_dict(
            values["policy.optimiserState"])
        reinforce.criticNet.load_state_dict(values["critic.net"]["state"])
        reinforce.criticOptimiser.load_state_dict(
            values["critic.optimiserState"])
        return reinforce

    @staticmethod
    def load(path: str) -> "A2C":
        """Load state from a file"""
        values = torch.load(path)
        return A2C.loadFromDict(values)

    def computeLosses(
            self, rewards: torch.Tensor, logProbs: torch.Tensor,
            predictedValues: torch.Tensor, isLive: torch.Tensor,
            discountFactor: float, lambdaa: float,
            entropyCoeff: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

        Returns:
            criticLoss: The critic loss for the minibatch.
            actorLoss: The actor loss for the minibatch.
        """
        T = len(rewards)
        advantages = torch.zeros(T, self.nEnvironments)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (rewards[t] +
                        discountFactor * isLive[t] * predictedValues[t + 1] -
                        predictedValues[t])
            gae = td_error + discountFactor * lambdaa * isLive[
                t] * gae  # type: ignore
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        criticLoss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actorLoss = (-(advantages.detach() * logProbs).mean() -
                     entropyCoeff * self.entropy.mean())
        return (criticLoss, actorLoss)

    def train(self, logger: Logger | None = None, **kwargs: Any) -> None:
        # Note that the async envs automatically terminate an episode
        # if it terminates or it truncates. So we don't need to manually
        # reset the environments.
        nEpochs = cast(int, kwargs["nEpochs"])
        nStepsPerEpoch = cast(int, kwargs["stepsPerEpoch"])

        checkpointFileName = cast(str, kwargs.get("checkpoints", ""))
        checkpointInterval = cast(int, kwargs.get("checkpointInterval", 100))

        discountFactor = cast(float, self._params["discountFactor"])
        assert discountFactor <= 1.0

        entropyCoeff = cast(float, self._params["entropyCoeff"])
        assert entropyCoeff <= 1.0
        nEnvironments = cast(int, self._params["nEnvironments"])

        lambdaa = cast(float, self._params["lambda"])
        assert lambdaa <= 1.0

        maxEpisodeLength = cast(int, self._params["maxEpisodeLength"])

        seed = cast(int, self._params["seed"])
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        rewarder = self.physicalModelSpec.rewarder(self._params)

        envFactories = []
        for eI in range(nEnvironments):
            envFactories.append(lambda: self.physicalModelSpec.makeEnv(
                params=self._params, max_episode_steps=maxEpisodeLength))
        envs = gym.vector.AsyncVectorEnv(envFactories)

        episodeLength = 0

        if logger:
            logger.startBlock()
            logger.log(
                f"[{datetime.datetime.now()}] : Training {nEpochs} epochs with {nStepsPerEpoch} steps per epoch, params are..."
            )
            logger.logDict(self._params)

        # at the start of training reset all envs to get an initial state
        states, info = envs.reset(seed=seed)

        self.actorNet.train()
        self.criticNet.train()

        # now train over our epochs
        for epoch in range(nEpochs):
            # reset lists that collect experiences of an episode (sample phase)
            predictedValues = torch.zeros(nStepsPerEpoch, nEnvironments)
            rewards = torch.zeros(nStepsPerEpoch, nEnvironments)
            logProbs = torch.zeros(nStepsPerEpoch, nEnvironments)
            isLive = torch.zeros(nStepsPerEpoch, nEnvironments)

            # gather data points for this epoch
            for t in range(nStepsPerEpoch):
                if self._abort:
                    if logger:
                        logger.log("Aborting computation!")
                    return
                # get our action and log prob from the actor
                tensorState = asTensor(states)
                action, logProb = self.actorNet.sample(tensorState)

                # get a value from the critic
                value = self.criticNet(tensorState)

                # move the environment, computing rewards
                states, reward, terminated, truncated, info = rewarder.step(
                    envs, action)

                predictedValues[t] = torch.squeeze(value)
                rewards[t] = torch.tensor(reward)
                logProbs[t] = logProb

                # add a mask (for the return calculation later);
                # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                terminatedL = cast(list, terminated)  # type : ignore

                isLive[t] = torch.tensor([not term for term in terminatedL])

            criticLoss, actorLoss = self.computeLosses(rewards, logProbs,
                                                       predictedValues, isLive,
                                                       discountFactor, lambdaa,
                                                       entropyCoeff)

            self.criticOptimiser.zero_grad()
            criticLoss.backward()
            self.criticOptimiser.step()

            self.actorOptimiser.zero_grad()
            actorLoss.backward()
            self.actorOptimiser.step()
            if checkpointFileName:
                if (epoch + 1) % checkpointInterval == 0:
                    fileName = f"{checkpointFileName}_{epoch + 1}.checkpoint"
                    self.save(fileName)

            if logger:
                if epoch % 10 == 0:
                    logger.startBlock()
                    logger.log(f"[{datetime.datetime.now()}] Epoch : { epoch}")
                    logger.log(f"Critic Loss : {criticLoss.detach().numpy()}")
                    logger.log(f"Actor Loss : {actorLoss.detach().numpy()}")
