from enum import StrEnum
from collections.abc import Sequence
from typing import Any, Callable, List, Optional, Protocol, Tuple
import numpy as np

from .params import ParamDict

import gymnasium as gym

__all__ = ["Rewarder", "PhysicalModelID", "PhysicalModelSpec"]


class PhysicalModelID(StrEnum):
    """Our choice of models to train."""
    invertedPendulum = "invertedPendulum"
    hopper = "hopper"
    halfCheetah = "halfCheetah"
    walker = "walker"


################################################################################
class Rewarder:
    """Base class that lets us override the default step function so
    we can define our own rewards/"""

    def preStep(self, env: Any, action: Any) -> None:
        """Called before step, to fetch out any state necessary for the reward computation.
            - env - the environment about to be stepped,
            - action - the action about to be taken
        """
        pass

    def postStep(self, env: Any, action: Any,
                 results: Tuple) -> Tuple[float, bool, bool]:
        """Called after step is called on the env, so our modifed reward computation can be
        called.
            - env - the environment that was just stepped,
            - action - the action just taken
            - results - the tuple returned by env.step

        Returns a tuple of...
            - float - the computed reward
            - bool  - whether the computation should terminate
            - bool  - whether the computation was truncated
        """
        observation, reward, terminated, truncated, info = results
        return (reward, terminated, truncated)

    def step(self, env: Any,
             action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.preStep(env, action)
        result = env.step(action)
        state, _, _, _, info = result
        reward, terminated, truncated = self.postStep(env, action, result)
        return (state, reward, terminated, truncated, info)


################################################################################
class PhysicalModelSpec(Protocol):
    """Protocol that defines aspects our physical model."""

    def makeEnv(self,
                *,
                params: ParamDict = {},
                **constructionArgs: Any) -> Any:
        """Retrieve the underlying mujoco env of this model"""
        ...

    def rewarder(self, params: ParamDict) -> Rewarder:
        """Retrieve the rewarder for this physical model.

        Initialise with the given params
        """
        ...

    def nActions(self) -> int:
        """How many values in actions for this model."""
        ...

    def actionRanges(self) -> Sequence[Tuple[float, float]]:
        """What is the valid range of each action."""

    def nObservations(self) -> int:
        """How many values in observations for this model."""
        ...

    def id(self) -> PhysicalModelID:
        """ID of this model."""
        ...

    def adjustScore(self, env: Any, score: float) -> float:
        """Adjust the score used for hyper parameter tuning."""
        ...
