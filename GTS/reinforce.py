from typing import Any, Dict, List, Optional, cast
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from enum import Enum

from .logger import Logger
from .params import ParamDict, bakeParams
from .parametricNet import ParametricNet, HiddenLayers
from .stochasticNet import StochasticNetLearnedSD, StochasticNetFixedSD, StochasticNet
from .physicalModel import PhysicalModelSpec
from .physicalModels import fetchPhysicalModelSpec
from .score import computeScore
from .tester import test
from .trajectory import Trajectory

__all__ = ["Reinforce", "Baseline"]


class Baseline(Enum):
    """What baseline to use for reinforce."""
    NONE = 0
    WHITE = 1


"""The params used by Reinforce and their defaults."""
_paramSpecs: ParamDict = {
    "policy.discountFactor": 0.99,
    "policy.nHidden": 2,
    "policy.hiddenSize": 32,
    "policy.hiddenActivation": "ReLU",
    "policy.learningRate": 1e-4,
    "baseline": "none",
    "seed": 1
}


class Reinforce:
    """Reinforce learning algorithm"""
    policyNet: StochasticNet

    def __init__(self, physicalModelSpec: PhysicalModelSpec,
                 params: ParamDict) -> None:
        """ Reinforcement algorithm to train a mujoco simulation
        :param : physicalModelSpec - the model we are training
        :param : params - a dict of hyper params
        """
        self._abort = False

        self.physicalModelSpec = physicalModelSpec
        self.env = physicalModelSpec.makeEnv()

        self._params = bakeParams(params, _paramSpecs)

        policyHiddenLayers = HiddenLayers.fromDict(self._params, "policy")

        if "policy.fixedVariance" in params:
            # fixed variance as a scale of the range of the environments?
            fixedVarianceFactor = self._params["policy.fixedVariance"]
            assert isinstance(fixedVarianceFactor, float)
            variances: List[float] = []
            for actionRange in physicalModelSpec.actionRanges():
                delta = actionRange[1] - actionRange[0]
                variances.append(delta * fixedVarianceFactor)
            self.policyNet = StochasticNetFixedSD(
                physicalModelSpec.nObservations(),
                physicalModelSpec.nActions(), policyHiddenLayers, variances)

        else:
            self.policyNet = StochasticNetLearnedSD(
                physicalModelSpec.nObservations(),
                physicalModelSpec.nActions(), policyHiddenLayers)

        self.discountFactor = cast(float,
                                   self._params["policy.discountFactor"])

        # Hyperparameters
        lr: float = cast(float, self._params["policy.learningRate"])
        self.policyOptimiser = torch.optim.AdamW(self.policyNet.parameters(),
                                                 lr=lr)

        # baseline
        baseline = self._params["baseline"]
        assert isinstance(baseline, str)
        baseline = baseline.lower()
        if baseline == "none":
            self.baselineType = Baseline.NONE
        elif baseline == "white":
            self.baselineType = Baseline.WHITE
        else:
            assert False, f"Unknown baseline type {baseline}"

        self._score: Dict[str, float] = {}

    def paramSpecs(self) -> ParamDict:
        return _paramSpecs

    def modelSpec(self) -> PhysicalModelSpec:
        return self.physicalModelSpec

    def policyNetwork(self) -> StochasticNet:
        return self.policyNet

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
        return "REINFORCE"

    def train(self, logger: Logger | None = None, **kwargs: Any) -> None:
        nSteps = cast(int, kwargs.get("nSteps", 50))

        if logger:
            logger.log(f"Training for {nSteps} steps, params are...")
            logger.logDict(self._params)

        self.policyNet.train()
        env = self.env

        seed = cast(int, self._params["seed"])

        rewarder = self.physicalModelSpec.rewarder(self._params)

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        bufferI: int = 0
        scores: np.ndarray = np.zeros(100)

        steps: int = 0
        episode: int = -1

        while steps < nSteps:
            episode += 1
            if self._abort:
                if logger:
                    logger.log("Aborting computation!")
                break

            run = Trajectory(seed, env, rewarder, self.policyNet, True)
            steps += run.nSteps
            scores[bufferI] = run.totalReward
            bufferI = (bufferI + 1) % 100

            # compute the discounted reward to go for each time step
            discountedReward: float = 0.0
            discountedRewards = torch.zeros((len(run.rewards)),
                                            dtype=torch.float32)
            for t in reversed(range(len(run.rewards))):
                thisReward = run.rewards[t]
                discountedReward = thisReward + self.discountFactor * discountedReward
                discountedRewards[t] = discountedReward

            if self.baselineType == Baseline.WHITE:
                # whitened baseline
                mean = discountedRewards.mean()
                std = discountedRewards.std()
                discountedRewards = (discountedRewards - mean) / (std + 1e-10)

            if self._abort:
                break

            # now compute the loss by multiplying the log probabilities by the the discounted rewards
            logProbByDiscounted: List[torch.Tensor] = []
            for logProb, reward in zip(run.logProbs, discountedRewards):
                logProbByDiscounted.append(logProb * reward)

            loss = -torch.stack(logProbByDiscounted).sum()

            # Update the policy network
            self.policyOptimiser.zero_grad()
            loss.backward()
            self.policyOptimiser.step()

            if episode > 99:
                score = computeScore(scores)
                if logger and (episode) % 100 == 0:
                    logger.logDict(
                        {
                            "steps": steps,
                            "score": round(score['score']),
                            "mean": round(score['mean']),
                            "COV": round(score['cov'], 3)
                        }, f"Episode {episode}, last 100 score...")

    def asDict(self) -> Dict[str, Any]:
        """Retrieve the dict needed for serialisation."""
        values: Dict[str, Any] = {
            "trainer": "Reinforce",
            "model": self.physicalModelSpec.id(),
            "params": self._params,
            "policy.net": self.policyNet.asDict(),
            "policy.optimiserState": self.policyOptimiser.state_dict(),
            "score": self._score,
        }
        return values

    def save(self, path: str) -> None:
        """Save this parameterised network out to the given file."""
        torch.save(self.asDict(), path)

    @staticmethod
    def loadFromDict(values: Dict[str, Any]) -> "Reinforce":
        """Load state from a serialisation dict"""
        assert values["trainer"] == "Reinforce"
        physicalModel = fetchPhysicalModelSpec(values["model"])
        params = values["params"]
        reinforce = Reinforce(physicalModel, params)
        reinforce.policyNet.load_state_dict(values["policy.net"]["state"])
        reinforce.policyOptimiser.load_state_dict(
            values["policy.optimiserState"])
        reinforce.setScore(values.get("score", {}))
        return reinforce

    @staticmethod
    def load(path: str) -> "Reinforce":
        """Load state from a file"""
        values = torch.load(path)
        return Reinforce.loadFromDict(values)
