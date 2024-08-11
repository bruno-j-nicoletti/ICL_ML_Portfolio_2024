from typing import Any, Dict, List

import gymnasium as gym
import torch
import numpy as np

from .score import computeScore
from .params import ParamDict
from .physicalModel import Rewarder, PhysicalModelSpec
from .stochasticNet import StochasticNet

__all__ = ["Trajectory"]


class Trajectory:
    """Runs a stochastic policy on a Gymnasium env, collecting data and computing a total reward for the run."""

    def __init__(self,
                 seed: int,
                 physicalModelEnv: Any,
                 rewarder: Rewarder,
                 policyNet: StochasticNet,
                 collectEverything: bool,
                 *,
                 deterministic: bool = False) -> None:
        rewards: List[float] = []
        self.actions: List[np.ndarray] = []
        self.logProbs: List[torch.Tensor] = []
        self.states: List[np.ndarray] = []
        self.nSteps: int = 0

        state, info = physicalModelEnv.reset(seed=seed)
        totalReward: float = 0.0
        while True:
            # get the distributions for our actions given the states
            if deterministic:
                action = policyNet.mean(state)
            else:
                action, logProb = policyNet.sample(state)
                logProb = logProb.sum()
                if collectEverything:
                    self.logProbs.append(logProb)

            # tell the rewarder the current state and the action we are about to take
            state, reward, terminated, truncated, info = rewarder.step(
                physicalModelEnv, action)
            totalReward += reward
            # record the reward and probability
            rewards.append(reward)
            if collectEverything:
                self.actions.append(action)
                self.states.append(state)

            if terminated or truncated:
                break

        self.rewards = np.array(rewards)
        self.totalReward = totalReward
        self.nSteps = len(self.rewards)
