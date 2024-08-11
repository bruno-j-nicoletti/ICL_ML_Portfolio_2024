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
    """Specification for our hopper."""

    def __init__(self) -> None:
        self.env = self.makeEnv()

    def makeEnv(self,
                *,
                params: ParamDict = {},
                **constructionArgs: Any) -> Any:
        return gym.make(_kModelName, **constructionArgs)

    def rewarder(self, params: ParamDict) -> Rewarder:
        return Rewarder()

    def nActions(self) -> int:
        return 6

    def actionRanges(self) -> Sequence[Tuple[float, float]]:
        return [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]

    def nObservations(self) -> int:
        return 17

    def id(self) -> PhysicalModelID:
        return PhysicalModelID.walker

    def adjustScore(self, env: Any, score: float) -> float:
        """Adjust the score used for hyper parameter tuning."""
        return score
        # attempted to modulate the reward by overall speed
        #xPos = env.unwrapped.data.qpos[0]
        #t = env._elapsed_steps

        # punish the score by the average speed
        #return score * xPos / t
