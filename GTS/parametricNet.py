import dataclasses
from collections.abc import Sequence
from typing import cast, Any, Callable, Dict, Optional, Tuple
from enum import StrEnum
import json

from .params import ParamDict
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal

from collections import OrderedDict

__all__ = ["ParametricNet", "Activation", "HiddenLayers", "asTensor"]


class Activation(StrEnum):
    """Our choice of activation functions."""
    tanh = "tanh"
    sigmoid = "sigmoid"
    ReLU = "ReLU"
    leakyReLU = "leakyReLU"
    ELU = "ELU"


Observations = np.ndarray | torch.Tensor


def asTensor(values: Observations) -> torch.Tensor:
    if isinstance(values, np.ndarray):
        return torch.tensor(values, dtype=torch.float32)
    else:
        return values


@dataclasses.dataclass
class HiddenLayers:
    nLayers: int
    size: int
    activation: Activation | str

    def toJSON(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def fromJSON(cls, s: str) -> "HiddenLayers":
        d = json.loads(s)
        return cls(**d)

    @staticmethod
    def fromDict(params: ParamDict, namespace: str) -> "HiddenLayers":
        nLayers = params[namespace + ".nHidden"]
        assert isinstance(nLayers, int)
        layerSize = params[namespace + ".hiddenSize"]
        assert isinstance(layerSize, int)
        layerActivation = params[namespace + ".hiddenActivation"]
        assert isinstance(layerActivation, str)
        return HiddenLayers(nLayers, layerSize, layerActivation)


################################################################################
class ParametricNet(nn.Module):
    """A Parameterised neural network."""

    def __init__(self,
                 nInputs: int,
                 nOutputs: int,
                 hiddenLayers: HiddenLayers,
                 *,
                 epsilon: float = 1e-6) -> None:
        """Make an NN, params are...
           - nInputs : the size of our observation vector
           - nOutputs : the size of our action vectors
           - hiddenLayers : specs for the hidden layers, for each one we have...
              - size
              - the activation function constructor for that layer
           - epsilon : small number for numerical stability
        """
        super(ParametricNet, self).__init__()

        self.epsilon = epsilon
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.hiddenLayers = hiddenLayers

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        lastLayer = nInputs
        for i in range(hiddenLayers.nLayers):
            layers[f"layer{i}"] = nn.Linear(lastLayer, hiddenLayers.size)
            if hiddenLayers.activation == Activation.tanh:
                layers[f"activation{i}"] = nn.Tanh()
            elif hiddenLayers.activation == Activation.sigmoid:
                layers[f"activation{i}"] = nn.Sigmoid()
            elif hiddenLayers.activation == Activation.ReLU:
                layers[f"activation{i}"] = nn.ReLU()
            elif hiddenLayers.activation == Activation.leakyReLU:
                layers[f"activation{i}"] = nn.LeakyReLU()
            elif hiddenLayers.activation == Activation.ELU:
                layers[f"activation{i}"] = nn.ELU()
            else:
                assert False, f"Unknow activation function {hiddenLayers.activation}"
            lastLayer = hiddenLayers.size
        self.commonNet = nn.Sequential(layers)

        # layer to compute the mean of our actions
        self.output = nn.Linear(lastLayer, nOutputs)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                n = m.in_features
                y = 1.0 / np.sqrt(n)
                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0)

    def forward(
            self, x: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        commonFeatures = self.commonNet(x)
        return self.output(commonFeatures)

    def asDict(self) -> Dict[str, Any]:
        """Retrieve the dict needed for serialisation."""
        return {
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
    def loadFromDict(values: Dict[str, Any]) -> "ParametricNet":
        """Create a new NN from a serialisation dict"""
        nInputs = values["nInputs"]
        nOutputs = values["nOutputs"]
        hiddenLayers = HiddenLayers.fromJSON(values["hiddenLayers"])
        epsilon = values.get("epsilon", 1e-6)
        net = ParametricNet(nInputs, nOutputs, hiddenLayers, epsilon=epsilon)
        net.load_state_dict(values["state"])
        return net

    @staticmethod
    def load(path: str) -> "ParametricNet":
        values = torch.load(path)
        return ParametricNet.loadFromDict(values)
