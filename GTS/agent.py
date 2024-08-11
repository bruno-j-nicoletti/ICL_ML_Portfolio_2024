from typing import Any, Dict, Protocol

from .logger import Logger
from .params import ParamDict
from .physicalModel import PhysicalModelSpec
from .stochasticNet import StochasticNet
from .trainingSpec import TrainingSpec

__all__ = ["Agent"]


class Agent(Protocol):

    def modelSpec(self) -> PhysicalModelSpec:
        """Get the model spec we are training."""
        ...

    def policyNetwork(self) -> StochasticNet:
        """Get the policy network for this agent."""
        ...

    def score(self) -> Dict[str, float]:
        """The score."""
        ...

    def setScore(self, score: Dict[str, float]) -> None:
        """Set the score."""
        ...

    def abort(self) -> None:
        """Stop processing."""
        ...

    def aborted(self) -> bool:
        """Was processing stopped."""
        ...

    def params(self) -> Dict[str, Any]:
        """Get the params used by this agent."""
        ...

    def asDict(self) -> Dict[str, Any]:
        """Retrieve the dict needed for serialisation."""
        ...

    def save(self, path: str) -> None:
        """Save this parameterised network out to the given file."""
        ...

    def train(self, logger: Logger | None = None, **kwargs: Any) -> None:
        """Train stuff"""
        ...

    def agentName(self) -> str:
        """Name of this agent."""
        ...

    def paramSpecs(self) -> ParamDict:
        """Get the set of params and their defaults for this agent."""
        ...
