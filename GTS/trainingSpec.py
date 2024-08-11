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
    """What to train. A model and the hyper params for it."""
    physicalModel: PhysicalModelID
    technique: str  # one of 'reinforce' or 'ppo'
    params: ParamDict = dataclasses.field(default_factory=dict)

    def toJSON(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    def save(self, path: str) -> None:
        with open(path, "wt") as f:
            f.write(self.toJSON())

    @classmethod
    def fromJSON(cls, s: str) -> "TrainingSpec":
        d = json.loads(s)
        return cls(**d)
