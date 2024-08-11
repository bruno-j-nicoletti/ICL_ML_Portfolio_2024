# PPO implmentation,
#
# heavily influenced by https://github.com/openai/spinningup.git
from typing import Any, Dict, List, Optional, Tuple, cast
import dataclasses
import random
import numpy as np
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

__all__ = ["PPO"]


def discountedSum(values: np.ndarray, discount: float) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float32)
    runningDiscount = 0.0
    for t in reversed(range(len(values))):
        runningDiscount = values[t] + discount * runningDiscount
        result[t] = runningDiscount
    return result


@dataclasses.dataclass
class TorchedPPOBuffer:
    """Our ppo observation buffer converted into tensors"""
    observations: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    logProbs: torch.Tensor


class PPOBuffer:
    """ A buffer for PPO trajectories."""

    def __init__(self, nObservations: int, nActions: int, size: int,
                 gamma: float, lambdaa: float) -> None:
        self.observations = np.zeros((size, nObservations), dtype=np.float32)
        self.actions = np.zeros((size, nActions), dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.logProbs = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lambdaa = lambdaa
        self.pos = 0
        self.pathStartIndex = 0
        self.max_size = size

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, val: float,
            logProb: float) -> None:
        """Add a timestep."""

        assert self.pos < self.max_size  # buffer has to have room so you can store
        self.observations[self.pos] = obs
        self.actions[self.pos] = act
        self.rewards[self.pos] = rew
        self.values[self.pos] = val
        self.logProbs[self.pos] = logProb
        self.pos += 1

    def endTrajectory(self, last_val: float = 0) -> float:
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).

        returns the discounted reward for the trajectory
        """

        path_slice = slice(self.pathStartIndex, self.pos)
        rewards = np.append(self.rewards[path_slice], last_val)
        values = np.append(self.values[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = discountedSum(deltas,
                                                    self.gamma * self.lambdaa)

        # the next line computes rewards-to-go, to be targets for the value function
        discountedReturns = discountedSum(rewards, self.gamma)
        self.returns[path_slice] = discountedReturns[:-1]
        self.pathStartIndex = self.pos
        return discountedReturns[0]

    def finish(self) -> TorchedPPOBuffer:
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.pos == self.max_size  # buffer has to be full before you can get
        self.pos, self.pathStartIndex = 0, 0
        # the next two lines implement the advantage normalization trick
        advantageMean = self.advantages.mean()
        #advantageSD = np.sqrt(((self.advantages - advantageMean)**2).sum()/len(self.advantages))
        advantageSD = np.std(self.advantages)
        self.advantages = (self.advantages - advantageMean) / advantageSD

        return TorchedPPOBuffer(
            torch.as_tensor(self.observations, dtype=torch.float32),
            torch.as_tensor(self.actions, dtype=torch.float32),
            torch.as_tensor(self.returns, dtype=torch.float32),
            torch.as_tensor(self.advantages, dtype=torch.float32),
            torch.as_tensor(self.logProbs, dtype=torch.float32))


"""The params used by PPO and their defaults."""
_paramSpecs: ParamDict = {
    "actor.learningRate": 1e-4,
    "actor.nHidden": 2,
    "actor.hiddenSize": 32,
    "actor.hiddenActivation": "ReLU",
    "actor.varianceScale": 0.2,
    "actor.trainingIterations": 80,
    "critic.learningRate": 1e-4,
    "critic.nHidden": 2,
    "critic.hiddenSize": 32,
    "critic.hiddenActivation": "ReLU",
    "critic.trainingIterations": 80,
    "clipRatio": 0.2,
    "targetKL": 0.01,
    "lambda": 0.97,
    "discountFactor": 0.99,
    "maxEpisodeLength": 1000,
    "seed": 1
}


################################################################################
class PPO:
    """PPO learning algorithm"""
    actorNet: StochasticNetFixedSD
    criticNet: ParametricNet

    def __init__(self, physicalModelSpec: PhysicalModelSpec,
                 params: ParamDict) -> None:
        """PPO algorithm to train a mujoco simulation
        """
        self._abort = False
        self._params = bakeParams(params, _paramSpecs)

        self.physicalModelSpec = physicalModelSpec
        self.env = physicalModelSpec.makeEnv()

        # make the actor network
        hiddenLayers = HiddenLayers.fromDict(self._params, "actor")
        fixedVarianceFactor = cast(float, self._params["actor.varianceScale"])
        variances: List[float] = []
        for actionRange in physicalModelSpec.actionRanges():
            delta = actionRange[1] - actionRange[0]
            variances.append(delta * fixedVarianceFactor)
        self.actorNet = StochasticNetFixedSD(physicalModelSpec.nObservations(),
                                             physicalModelSpec.nActions(),
                                             hiddenLayers, variances)
        lr: float = cast(float, self._params["actor.learningRate"])
        self.actorOptimiser = torch.optim.AdamW(self.actorNet.parameters(),
                                                lr=lr)

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
        return "PPO"

    def asDict(self) -> Dict[str, Any]:
        """Retrieve the dict needed for serialisation."""
        values: Dict[str, Any] = {
            "trainer": "PPO",
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
    def loadFromDict(values: Dict[str, Any]) -> "PPO":
        """Load state from a serialisation dict"""
        assert values["trainer"] == "PPO"
        physicalModel = fetchPhysicalModelSpec(values["model"])
        params = values["params"]
        reinforce = PPO(physicalModel, params)
        reinforce.actorNet.load_state_dict(values["policy.net"]["state"])
        reinforce.actorOptimiser.load_state_dict(
            values["policy.optimiserState"])
        reinforce.criticNet.load_state_dict(values["critic.net"]["state"])
        reinforce.criticOptimiser.load_state_dict(
            values["critic.optimiserState"])
        return reinforce

    @staticmethod
    def load(path: str) -> "PPO":
        """Load state from a file"""
        values = torch.load(path)
        return PPO.loadFromDict(values)

    def train(self, logger: Logger | None = None, **kwargs: Any) -> None:
        nEpochs = cast(int, kwargs["nEpochs"])
        stepsPerEpoch = cast(int, kwargs["stepsPerEpoch"])
        self.actorNet.train()
        self.criticNet.train()

        checkpointFileName = cast(str, kwargs.get("checkpoints", ""))
        checkpointInterval = cast(int, kwargs.get("checkpointInterval", 100))

        seed = cast(int, self._params["seed"])
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        clipRatio = cast(float, self._params["clipRatio"])

        discountFactor = cast(float, self._params["discountFactor"])
        assert discountFactor <= 1.0

        lambdaa = cast(float, self._params["lambda"])
        assert lambdaa <= 1.0

        maxEpisodeLength = cast(int, self._params["maxEpisodeLength"])
        assert isinstance(maxEpisodeLength, int)

        actorTrainingIterations = cast(
            int, self._params["actor.trainingIterations"])

        criticTrainingIterations = cast(
            int, self._params["critic.trainingIterations"])

        # Kullback–Leibler divergence
        targetKL = cast(float, self._params["targetKL"])

        env = self.env
        rewarder = self.physicalModelSpec.rewarder(self._params)

        ppoBuffer = PPOBuffer(self.physicalModelSpec.nObservations(),
                              self.physicalModelSpec.nActions(), stepsPerEpoch,
                              discountFactor, lambdaa)

        lastState, info = env.reset(seed=seed)
        episodeLength = 0

        if logger:
            logger.startBlock()
            logger.log(
                f"Training {nEpochs} epochs with {stepsPerEpoch} steps per epoch, params are..."
            )
            logger.logDict(self._params)

        def computeActorLoss(data: TorchedPPOBuffer) -> Tuple:
            distribution = self.actorNet.distribution(data.observations)
            newLogProbs = distribution.log_prob(data.actions).sum(axis=-1)
            ratio = torch.exp(newLogProbs - data.logProbs)
            clippedAdvantages = torch.clamp(ratio, 1 - clipRatio,
                                            1 + clipRatio) * data.advantages
            loss = -(torch.min(ratio * data.advantages,
                               clippedAdvantages)).mean()

            klApprox = (data.logProbs - newLogProbs).mean().item()
            entropy = distribution.entropy().mean().item()
            wasClipped = ratio.gt(1 + clipRatio) | ratio.lt(1 - clipRatio)
            clipFraction = torch.as_tensor(wasClipped,
                                           dtype=torch.float32).mean().item()

            return loss, klApprox, entropy, clipFraction

        def trainNetworks(data: TorchedPPOBuffer) -> None:
            oldActorLoss: float | None = None
            # train actor
            for _ in range(actorTrainingIterations):
                if self._abort:
                    return
                self.actorOptimiser.zero_grad()
                actorLoss, klApprox, entropy, clipFraction = computeActorLoss(
                    data)
                if not oldActorLoss:
                    oldActorLoss = actorLoss.item()

                if targetKL > 0.0 and klApprox > 1.5 * targetKL:
                    if logger:
                        logger.log(
                            f"Stopping actor training on {_} because KL failure. {klApprox}"
                        )
                    break
                actorLoss.backward()
                self.actorOptimiser.step()

            # train critic
            oldCriticLoss: float | None = None
            for i in range(criticTrainingIterations):
                if self._abort:
                    return
                self.criticOptimiser.zero_grad()
                criticLoss = ((self.criticNet(data.observations) -
                               data.returns)**2).mean()
                if not oldCriticLoss:
                    oldCriticLoss = criticLoss.item()
                criticLoss.backward()
                self.criticOptimiser.step()

        scoresI: int = 0
        scores: np.ndarray = np.zeros(100)
        nEpisodes = 0

        steps = 0
        for epoch in range(nEpochs):
            if logger:
                logger.startBlock()
                logger.log(f"Epoch : { epoch}")
            # gather data points for this epoch
            for t in range(stepsPerEpoch):

                if self._abort:
                    if logger:
                        logger.log("Aborting computation!")
                    return
                # get our action and log prob from the actor
                tensorState = asTensor(lastState)
                action, logProb = self.actorNet.sample(tensorState)

                # get a value from the critic
                value = self.criticNet(tensorState)

                # move the environment, computing rewards
                state, reward, terminated, truncated, info = rewarder.step(
                    env, action)

                episodeLength += 1
                steps += 1
                # record it
                ppoBuffer.add(lastState, action, reward, value,
                              logProb.detach().numpy())
                lastState = state

                maxedOutEpisodeLenth = episodeLength == maxEpisodeLength
                epochEnding = t == stepsPerEpoch - 1

                if terminated or maxedOutEpisodeLenth or epochEnding:
                    if maxedOutEpisodeLenth or epochEnding:
                        value = self.criticNet(
                            asTensor(lastState)).detach().numpy()
                    else:
                        value = 0.0
                    reward = ppoBuffer.endTrajectory(value)
                    episodeLength = 0
                    lastState, _ = env.reset()

                    # only include full episodes in the score
                    if logger:  #and not epochEnding:
                        scores[scoresI] = reward
                        scoresI = (scoresI + 1) % 100
                        nEpisodes += 1

                        if nEpisodes % 100 == 0 and nEpisodes > 0:
                            score = computeScore(scores)
                            logger.logDict(
                                {
                                    "steps": steps,
                                    "score": round(score['score']),
                                    "mean": round(score['mean']),
                                    "COV": round(score['cov'], 3)
                                }, f"Episode {nEpisodes}, last 100 score...")

            # now train it
            trainNetworks(ppoBuffer.finish())

            if checkpointFileName:
                if (epoch + 1) % checkpointInterval == 0:
                    fileName = f"{checkpointFileName}_{epoch + 1}.checkpoint"
                    self.save(fileName)
