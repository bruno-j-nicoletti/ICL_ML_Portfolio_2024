import optuna

from enum import Enum
import itertools
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Sequence

__all__ = ["ParamType", "ParamDict", "printParams"]

################################################################################
# what a parameter can be
ParamType = Union[int, float, str]

################################################################################
# A set of parameters
ParamDict = Dict[str, ParamType]


################################################################################
def printParams(params: ParamDict | None) -> None:
    # dump params to stdout
    if params:
        print(f"Params...")
        for k, v in params.items():
            print(f"   {k} : {v}")


################################################################################
def bakeParams(params: ParamDict, paramSpecs: ParamDict) -> ParamDict:
    """
    :param: params -  the parameters that have been setf by a used
    :param: paramSpecs - the params we are need and defaults values

    Generate a set of parameters, where if a value is not in params but is found
    it paramSpecs, that will be used as a default
    """
    result: ParamDict = {}
    for k, defaultV in paramSpecs.items():
        v = params.get(k)
        if v is not None:
            assert isinstance(v, type(defaultV))
            result[k] = v
        else:
            result[k] = defaultV
    for k, v in params.items():
        if k not in result:
            result[k] = v
    return result
