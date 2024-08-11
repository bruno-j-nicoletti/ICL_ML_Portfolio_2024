from typing import cast, Any, Dict, List, Optional, Protocol, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal

from .parametricNet import ParametricNet, HiddenLayers, Observations, asTensor

__all__ = [
    "StochasticNet", "StochasticNetLearnedSD", "loadStochasticNetFromDict"
]


class StochasticNet(ParametricNet):
    """Base class for a stochastic neural network."""

    def sample(self,
               observations: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Sample a value from the distribution given the observation.

        Returns the sampled value and the log probability."""
        raise NotImplementedError

    def distribution(
        self, observations: np.ndarray | torch.Tensor
    ) -> torch.distributions.Distribution:
        """Get the distribution for the given the observations."""
        raise NotImplementedError

    def mean(self, observations: np.ndarray) -> np.ndarray:
        """Return the mean of the computed distribution."""
        raise NotImplementedError

    def asDict(self) -> Dict[str, Any]:
        """Retrieve the dict needed for serialisation."""
        raise NotImplementedError


################################################################################
class StochasticNetLearnedSD(StochasticNet):
    """A Parameterised stochastic network that generates probability distributions where
    the standard deviations are learned."""

    def __init__(self,
                 nInputs: int,
                 nOutputs: int,
                 hiddenLayers: HiddenLayers,
                 *,
                 epsilon: float = 1e-6) -> None:
        """Make an NN, params are...
           - nInputs : the size of our observartion vector
           - nOutputs : the size of our action vectors
           - hiddenLayers : specs for the hidden layers, for each one we have...
              - size
              - the activation function constructor for that layer
           - epsilon : small number for numerical stability
        """
        super().__init__(nInputs, nOutputs, hiddenLayers, epsilon=epsilon)
        self.stdDev = nn.Linear(hiddenLayers.size, nOutputs)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        commonFeatures = self.commonNet(x)
        means = self.output(commonFeatures)
        stdDev = torch.exp(self.stdDev(commonFeatures))
        return (means, stdDev)

    def distribution(self, observations: Observations) -> Normal:
        """Compute the distribution for the given the observations."""
        means, stddevs = self(asTensor(observations))
        return Normal(means + self.epsilon, stddevs + self.epsilon)

    def sample(self,
               observations: Observations) -> Tuple[np.ndarray, torch.Tensor]:
        """Sample a value from the distribution given the observation.

        Returns the value and the log probability."""
        distrib = self.distribution(observations)
        value = distrib.sample()
        logProb = distrib.log_prob(value).sum()
        value = value.reshape(self.nOutputs)
        return value.detach().numpy(), logProb

    def mean(self, observations: Observations) -> np.ndarray:
        """Return the mean of the computed distribution."""
        means, _ = self(asTensor(observations))
        return means.detach().numpy()

    def asDict(self) -> Dict[str, Any]:
        """Retrieve the dict needed for serialisation."""
        return {
            "class": "StochasticNetLearnedSD",
            "state": self.state_dict(),
            "nInputs": self.nInputs,
            "nOutputs": self.nOutputs,
            "hiddenLayers": self.hiddenLayers.toJSON(),
            "epsilon": self.epsilon,
        }

    def save(self, path: str) -> None:
        """Save this parameterised network out to the given file."""
        torch.save(self.asDict(), path)

    @staticmethod
    def loadFromDict(values: Dict[str, Any]) -> "StochasticNetLearnedSD":
        """Create a new NN from a serialisation dict"""
        nInputs = values["nInputs"]
        nOutputs = values["nOutputs"]
        hiddenLayers = HiddenLayers.fromJSON(values["hiddenLayers"])
        epsilon = values.get("epsilon", 1e-6)
        net = StochasticNetLearnedSD(nInputs,
                                     nOutputs,
                                     hiddenLayers,
                                     epsilon=epsilon)
        net.load_state_dict(values["state"])
        return net

    @staticmethod
    def load(path: str) -> "ParametricNet":
        """Load it from a path"""
        values = torch.load(path)
        return ParametricNet.loadFromDict(values)


################################################################################
class StochasticNetFixedSD(StochasticNet):
    """A Parameterised stochastic network that generates probability distributions where
    the standard deviations are specified."""

    def __init__(self,
                 nInputs: int,
                 nOutputs: int,
                 hiddenLayers: HiddenLayers,
                 variance: float | List[float] | List[List[float]],
                 *,
                 epsilon: float = 1e-6) -> None:
        """Make an NN, params are...
           - nInputs : the size of our observartion vector
           - nOutputs : the size of our action vectors
           - hiddenLayers : specs for the hidden layers, for each one we have...
              - size
              - the activation function constructor for that layer
           - epsilon : small number for numerical stability

        """
        super().__init__(nInputs, nOutputs, hiddenLayers, epsilon=epsilon)

        if isinstance(variance, float):
            self.variance = torch.full(size=(nOutputs, ),
                                       fill_value=variance,
                                       dtype=torch.float32)
        else:
            assert isinstance(variance, list)
            self.variance = torch.tensor(variance, dtype=torch.float32)
        self.varianceValue = variance  # for serialisation

    def distribution(self, observations: Observations) -> Normal:
        """Compute the distribution for the given the observations."""
        means = self(asTensor(observations))
        return Normal(means, self.variance)

    def sample(self,
               observations: Observations) -> Tuple[np.ndarray, torch.Tensor]:
        """Sample a value from the distribution given the observation.

        Returns the value and the log probability."""
        distrib = self.distribution(observations)
        value = distrib.sample()
        logProb = distrib.log_prob(value).sum()
        #value = value.reshape(self.nOutputs)
        return value.detach().numpy(), logProb

    def mean(self, observations: Observations) -> np.ndarray:
        """Return the mean of the computed distribution."""
        means = self(asTensor(observations))
        return means.detach().numpy()

    def asDict(self) -> Dict[str, Any]:
        """Retrieve the dict needed for serialisation."""
        return {
            "class": "StochasticNetFixedSD",
            "state": self.state_dict(),
            "nInputs": self.nInputs,
            "nOutputs": self.nOutputs,
            "hiddenLayers": self.hiddenLayers.toJSON(),
            "epsilon": self.epsilon,
            "variance": self.varianceValue
        }

    def save(self, path: str) -> None:
        """Save this parameterised network out to the given file."""
        torch.save(self.asDict(), path)

    @staticmethod
    def loadFromDict(values: Dict[str, Any]) -> "StochasticNetFixedSD":
        """Create a new NN from a serialisation dict"""
        nInputs = values["nInputs"]
        nOutputs = values["nOutputs"]
        hiddenLayers = HiddenLayers.fromJSON(values["hiddenLayers"])
        variance = values["variance"]
        epsilon = values.get("epsilon", 1e-6)
        net = StochasticNetFixedSD(nInputs,
                                   nOutputs,
                                   hiddenLayers,
                                   variance,
                                   epsilon=epsilon)
        net.load_state_dict(values["state"])
        return net

    @staticmethod
    def load(path: str) -> "ParametricNet":
        """Load it from a path"""
        values = torch.load(path)
        return ParametricNet.loadFromDict(values)


################################################################################
def loadStochasticNetFromDict(values: Dict[str, Any]) -> "StochasticNet":
    """Load a stochastic network from a dict"""
    theClass = values.get("class", "StochasticNetLearnedSD")
    if theClass == "StochasticNetLearnedSD":
        return StochasticNetLearnedSD.loadFromDict(values)
    elif theClass == "StochasticNetFixedSD":
        return StochasticNetFixedSD.loadFromDict(values)
    else:
        assert False, f"Unknown type of stochatic net '{theClass}'"
