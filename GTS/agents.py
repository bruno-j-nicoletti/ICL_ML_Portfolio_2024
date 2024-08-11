from typing import Any, Dict, Protocol

from .physicalModel import PhysicalModelSpec
from .physicalModels import fetchPhysicalModelSpec
from .PPO import PPO
from .A2C import A2C
from .reinforce import Reinforce
from .stochasticNet import StochasticNet
from .trainingSpec import TrainingSpec
from .agent import Agent
import torch

__all__ = ["makeAgentFromSpec", "loadAgentFromFile", "loadAgentFromDict"]


def makeAgentFromSpec(trainingSpec: TrainingSpec) -> Agent:
    """Create an agent from a training spec."""
    modelSpec = fetchPhysicalModelSpec(trainingSpec.physicalModel)
    params = trainingSpec.params
    agent: Agent
    if trainingSpec.technique.lower() == "reinforce":
        return Reinforce(modelSpec, params)
    elif trainingSpec.technique.lower() == "ppo":
        return PPO(modelSpec, params)
    elif trainingSpec.technique.lower() == "a2c":
        return A2C(modelSpec, params)
    else:
        assert False, f"Unknown training technique {trainingSpec.technique}"


def loadAgentFromDict(values: Dict[str, Any]) -> Agent:
    """Load an agent from a dict that was previously serialised."""
    agentName = values.get("agent", values["trainer"]).lower()
    if agentName == "reinforce":
        return Reinforce.loadFromDict(values)
    elif agentName == "ppo":
        return PPO.loadFromDict(values)
    elif agentName == "a2c":
        return A2C.loadFromDict(values)
    else:
        assert False, f"Unknown agent technique {agentName}"


def loadAgentFromFile(path: str) -> Agent:
    """Load an agent from a file."""
    values = torch.load(path)
    return loadAgentFromDict(values)
