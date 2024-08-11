from collections.abc import Sequence
from typing import Any, List, Tuple, cast
import numpy as np
import gymnasium as gym

from .physicalModel import Rewarder, PhysicalModelSpec, PhysicalModelID
from .params import ParamDict

__all__ = ["HalfCheetahSpec"]

# name of our model
_kModelName = "HalfCheetah-v4"


################################################################################
class HalfCheetahSpec(PhysicalModelSpec):
    """Specification for the half cheetah."""

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
        return PhysicalModelID.halfCheetah

    def adjustScore(self, env: Any, score: float) -> float:
        """Adjust the score used for hyper parameter tuning."""
        return score
