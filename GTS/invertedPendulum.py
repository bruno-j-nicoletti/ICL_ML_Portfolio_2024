from collections.abc import Sequence
from typing import Any, List, Tuple
import numpy as np

import gymnasium as gym

from .physicalModel import Rewarder, PhysicalModelSpec, PhysicalModelID
from .params import *

__all__ = ["InvertedPendulumSpec"]

_SPEED_PENALTY = "InvertedPendulum/speedPenalty"


################################################################################
class InvertedPendulumRewarder(Rewarder):
    """Reward computer."""

    def __init__(self, params: ParamDict) -> None:
        # grab hyper params that affect our reward
        self.speedPenalty = params.get(_SPEED_PENALTY, 0)

    def postStep(self, env: Any, action: np.ndarray,
                 result: Tuple) -> Tuple[float, bool, bool]:
        # tweak rewards after the environment has been stepped
        observation, reward, terminated, truncated, info = result

        # punish the cart moving too fast, which is obs[2]
        if len(observation.shape) > 1:
            speeds = np.array([s[2] for s in observation])
            reward = reward - abs(speeds * self.speedPenalty)
        else:
            reward = reward - abs(observation[2]) * self.speedPenalty

        return (reward, terminated, truncated)


_kModelName = "InvertedPendulum-v4"


################################################################################
class InvertedPendulumSpec(PhysicalModelSpec):
    """Specification for our inverted pendulum."""

    def __init__(self) -> None:
        self.env = self.makeEnv()

    def makeEnv(self,
                *,
                params: ParamDict = {},
                **constructionArgs: Any) -> Any:
        # make an environment with the given hyper params and optional constuction args
        return gym.make(_kModelName, **constructionArgs)

    def rewarder(self, params: ParamDict) -> InvertedPendulumRewarder:
        """Override the step func so we can get access to the rewards."""
        return InvertedPendulumRewarder(params)

    def nActions(self) -> int:
        """How many values in actions for this model."""
        return self.env.action_space.shape[0]

    def actionRanges(self) -> Sequence[Tuple[float, float]]:
        """What is the valid range of each action."""
        return [(-3, 3)]

    def nObservations(self) -> int:
        """How many values in observations for this model."""
        return self.env.observation_space.shape[0]

    def id(self) -> PhysicalModelID:
        """ID of this model."""
        return PhysicalModelID.invertedPendulum

    def adjustScore(self, env: Any, score: float) -> float:
        """Adjust the score used for hyper parameter tuning."""
        return score
