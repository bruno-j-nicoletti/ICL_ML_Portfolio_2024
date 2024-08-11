from typing import Any, Dict, List, Optional

import signal
import cv2
import torch
import numpy as np
from gymnasium.wrappers import HumanRendering, RecordVideo

from .score import computeScore
from .params import ParamDict
from .physicalModel import Rewarder, PhysicalModelSpec
from .stochasticNet import StochasticNet

__all__ = ["play"]


def play(physicalModelSpec: PhysicalModelSpec,
         policy: StochasticNet,
         *,
         stochastic: bool = False,
         seed: int = 1,
         videoFileName: Optional[str] = None,
         nFrames: int = 1000,
         loops: int = 2,
         breakOnTermination: bool = False) -> None:

    policy.eval()

    if videoFileName == "":
        videoFileName = None

    if videoFileName is None:
        env = physicalModelSpec.makeEnv(render_mode="human")
    else:
        env = physicalModelSpec.makeEnv(render_mode="rgb_array")

    video: Optional[cv2.VideoWriter] = None

    print(f"Framestep is {env.unwrapped.dt}s")
    obs, info = env.reset(seed=seed)
    i: int = 0
    loop = 0
    died = False
    while True:
        # have the net sample the action, along with it's probability
        if stochastic:
            action, _ = policy.sample(obs)
        else:
            action = policy.mean(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        i = i + 1
        if videoFileName is not None:
            frame = env.render()
            if video is None:
                frameRate = env.metadata['render_fps']
                assert isinstance(frameRate, int)
                video = cv2.VideoWriter(
                    videoFileName,
                    cv2.VideoWriter_fourcc(*'mp4v'),  # type: ignore
                    frameRate,
                    (frame.shape[1], frame.shape[0]))
            video.write(frame)
            if i == nFrames:
                break
        else:
            if terminated and not died:
                print("Died on frame ", i)
                died = True
            if i == nFrames or (terminated and breakOnTermination):
                if loops > 0 and loop + 1 == loops:
                    break
                i = 0
                loop += 1
                obs, info = env.reset()

    if videoFileName is not None:
        assert video is not None
        video.release()
