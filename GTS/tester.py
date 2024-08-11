from typing import Any, Dict, List

import gymnasium as gym
import torch
import numpy as np

from .score import computeScore
from .params import ParamDict
from .physicalModel import Rewarder, PhysicalModelSpec
from .stochasticNet import StochasticNet
from .trajectory import Trajectory

__all__ = ["test"]


def test(physicalModelSpec: PhysicalModelSpec,
         policyNet: StochasticNet,
         params: ParamDict,
         nEpisodes: int,
         nFrames: int,
         deterministic: bool = True) -> Dict[str, float]:
    """Test the policy net and return a score."""
    # make a clean environment and rewarder
    env = physicalModelSpec.makeEnv()
    rewarder = physicalModelSpec.rewarder(params)
    policyNet.eval()

    seed = params.get("seed", 1)
    assert isinstance(seed, int)
    rewards: List[float] = []
    # run it for the given number of episodes
    for e in range(nEpisodes):
        run = Trajectory(seed + e,
                         env,
                         rewarder,
                         policyNet,
                         collectEverything=False,
                         deterministic=deterministic)
        reward = physicalModelSpec.adjustScore(env, run.totalReward)
        rewards.append(reward)
    # compute a score from all the trajectories
    return computeScore(rewards)
