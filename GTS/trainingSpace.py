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
# The dict that defines hyper params to search ove
# each entry is either..
#   - a list of values, which will be used categorially
#   - a dict contain "min", "max" and optionaly "log" (True|False) and "step"
ParamSpace = Dict[str, List | Dict[str, ParamType | bool]]


################################################################################
@dataclasses.dataclass
class TrainingSpace:
    """How to train an agent over a given paramter space."""
    physicalModel: PhysicalModelID  # id of the model
    name: str  # name of this training space and corresponding optuna study
    technique: str  # one of 'reinforce' or 'ppo'
    nSteps: int  # the number of training steps to run
    nEpochs: int = 50  # the number of training epochs to run
    params: ParamDict = dataclasses.field(default_factory=dict)  # fixed params
    paramSpace: ParamSpace = dataclasses.field(
        default_factory=dict)  # param space

    def bake(self, trial: optuna.trial.Trial) -> TrainingSpec:
        """
        Sample our hyper parameter space from the optuna trial and combine those
        with the fixed hyper parameters to make a TrainingSpec
        """
        bakedParams: ParamDict = copy.deepcopy(self.params)
        for pName, pSpec in self.paramSpace.items():
            if isinstance(pSpec, list):
                # list is categorical
                bakedParams[pName] = trial.suggest_categorical(pName, pSpec)
            elif isinstance(pSpec, dict):
                # dict is a range
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
        # save to a JSON string
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def fromJSON(cls, s: str) -> "TrainingSpace":
        # load from a JSON string
        d = json.loads(s)
        return cls(**d)
