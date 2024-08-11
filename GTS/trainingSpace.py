import copy
import dataclasses
import json
import optuna
from typing import Callable, Dict, List, Tuple

from .params import ParamType, ParamDict
from .trainingSpec import TrainingSpec
from .physicalModel import PhysicalModelID

__all__ = ["TrainingSpace"]

################################################################################
# the space for all our hyper parameters
# if a list, it is categorical, for a
ParamSpace = Dict[str, List | Dict[str, ParamType | bool]]


################################################################################
@dataclasses.dataclass
class TrainingSpace:
    """How to train an agent over a given paramter space."""
    physicalModel: PhysicalModelID
    name: str  # name of this training pass
    technique: str  # one of 'reinforce' or 'ppo'
    nSteps: int  # the number of training steps to run
    nEpochs: int = 50  # the number of training epochs to run
    params: ParamDict = dataclasses.field(default_factory=dict)  # fixed params
    paramSpace: ParamSpace = dataclasses.field(
        default_factory=dict)  # param space

    def bake(self, trial: optuna.trial.Trial) -> TrainingSpec:
        """Bake our param space into a single set of hyper params for the given trial."""
        bakedParams: ParamDict = copy.deepcopy(self.params)
        for pName, pSpec in self.paramSpace.items():
            if isinstance(pSpec, list):
                bakedParams[pName] = trial.suggest_categorical(pName, pSpec)
            elif isinstance(pSpec, dict):
                step: float | int | None
                mini: float | int = pSpec["min"]  # type: ignore
                maxi: float | int = pSpec["max"]  # type: ignore
                isLog: bool = pSpec.get("log", False) == True
                if isinstance(mini, int):
                    step = pSpec.get("step", 1)  # type: ignore
                    assert isinstance(maxi, int)
                    assert isinstance(step, int)
                    bakedParams[pName] = trial.suggest_int(pName,
                                                           mini,
                                                           maxi,
                                                           step=step,
                                                           log=isLog)
                else:
                    assert isinstance(mini, float)
                    assert isinstance(maxi, float)
                    step = pSpec.get("step", None)  # type: ignore
                    assert isinstance(step, float) or step is None
                    bakedParams[pName] = trial.suggest_float(pName,
                                                             mini,
                                                             maxi,
                                                             step=step,
                                                             log=isLog)

        return TrainingSpec(self.physicalModel, self.technique, bakedParams)

    def toJSON(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def fromJSON(cls, s: str) -> "TrainingSpace":
        d = json.loads(s)
        return cls(**d)
