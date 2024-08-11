from collections.abc import Sequence
from typing import Any, List, Tuple, cast
import numpy as np
import gymnasium as gym

from .physicalModel import Rewarder, PhysicalModelSpec, PhysicalModelID
from .params import ParamDict

__all__ = ["WalkerSpec"]

# name of our model
_kModelName = "Walker2d-v4"


################################################################################
class WalkerSpec(PhysicalModelSpec):
    """Specification for the walker model."""

    def __init__(self) -> None:
        self.env = self.makeEnv()

    def makeEnv(self,
                *,
                params: ParamDict = {},
                **constructionArgs: Any) -> Any:
        # make an environment with the given hyper params and optional constuction args
        return gym.make(_kModelName, **constructionArgs)

    def rewarder(self, params: ParamDict) -> Rewarder:
        # make the defautl rewarder
        return Rewarder()

    def nActions(self) -> int:
        # number of actions
        return 6

    def actionRanges(self) -> Sequence[Tuple[float, float]]:
        # ranges of those actions
        return [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]

    def nObservations(self) -> int:
        # number of observations
        return 17

    def id(self) -> PhysicalModelID:
        # id of this physical model
        return PhysicalModelID.walker

    def adjustScore(self, env: Any, score: float) -> float:
        """Adjust the score used for hyper parameter tuning."""
        return score
        # attempted to modulate the reward by overall speed
        #xPos = env.unwrapped.data.qpos[0]
        #t = env._elapsed_steps

        # punish the score by the average speed
        #return score * xPos / t
