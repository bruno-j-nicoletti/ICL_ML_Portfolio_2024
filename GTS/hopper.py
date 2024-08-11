from collections.abc import Sequence
from typing import Any, List, Tuple, cast
import numpy as np
import gymnasium as gym

from .physicalModel import Rewarder, PhysicalModelSpec, PhysicalModelID
from .params import ParamDict

__all__ = ["HopperSpec"]

_FORWARD_REWARD_WEIGHT = "Hopper/forwardRewardWeight"
_FORWARD_REWARD_MINIMUM = "Hopper/minimumSpeed"
_CONTROL_COST_WEIGHT = "Hopper/controlPenalty"

# name of our model
_kModelName = "Hopper-v4"


################################################################################
class HopperRewarder(Rewarder):
    """Reward computer."""

    def __init__(self, params: ParamDict) -> None:
        self.forwardRewardWeight = cast(
            float, params.get(_FORWARD_REWARD_WEIGHT, 1.0))
        self.minimumSpeed = cast(float, params.get(_FORWARD_REWARD_MINIMUM,
                                                   0.0))
        self.ctrlCostWeight = cast(float,
                                   params.get(_CONTROL_COST_WEIGHT, 0.001))

    def postStep(self, env: Any, action: np.ndarray,
                 result: Tuple) -> Tuple[float, bool, bool]:
        observation, _, terminated, truncated, info = result

        speed = info['x_velocity']
        speed -= self.minimumSpeed
        speed *= self.forwardRewardWeight

        if isinstance(terminated, np.ndarray):
            healthy = [float(not dead) for dead in terminated]
        else:
            healthy = float(not terminated)  # type: ignore
        reward = speed + healthy - np.sum(
            np.square(action)) * self.ctrlCostWeight

        return (reward, terminated, truncated)


################################################################################
class HopperSpec(PhysicalModelSpec):
    """Specification for our hopper."""

    def __init__(self) -> None:
        self.env = self.makeEnv()

    def makeEnv(self,
                *,
                params: ParamDict = {},
                **constructionArgs: Any) -> Any:
        return gym.make(_kModelName, **constructionArgs)

    def rewarder(self, params: ParamDict) -> HopperRewarder:
        return HopperRewarder(params)

    def nActions(self) -> int:
        return self.env.action_space.shape[0]

    def actionRanges(self) -> Sequence[Tuple[float, float]]:
        return [(-1, 1), (-1, 1), (-1, 1)]

    def nObservations(self) -> int:
        return self.env.observation_space.shape[0]

    def id(self) -> PhysicalModelID:
        return PhysicalModelID.hopper

    def adjustScore(self, env: Any, score: float) -> float:
        """Adjust the score used for hyper parameter tuning."""
        return score
        # attempted to modulate the reward by overall speed
        #xPos = env.unwrapped.data.qpos[0]
        #t = env._elapsed_steps

        # punish the score by the average speed
        #return score * xPos / t
