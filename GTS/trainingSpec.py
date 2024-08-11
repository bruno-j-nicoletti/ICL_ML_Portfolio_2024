import copy
import dataclasses
import json
import optuna
from typing import Callable, Dict, List, Tuple

from .params import ParamType, ParamDict
from .physicalModel import PhysicalModelID

__all__ = ["TrainingSpec"]


################################################################################
@dataclasses.dataclass
class TrainingSpec:
    """How to train a model"""
    physicalModel: PhysicalModelID  # the id of the model being traied
    technique: str  # one of 'reinforce' or 'ppo'
    params: ParamDict = dataclasses.field(default_factory=dict)  # hyper params

    def toJSON(self) -> str:
        # save to a json string
        return json.dumps(dataclasses.asdict(self), indent=2)

    def save(self, path: str) -> None:
        # save to a file
        with open(path, "wt") as f:
            f.write(self.toJSON())

    @classmethod
    def fromJSON(cls, s: str) -> "TrainingSpec":
        # load from a json string
        d = json.loads(s)
        return cls(**d)
