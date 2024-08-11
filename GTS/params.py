import optuna

from enum import Enum
import itertools
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Sequence

__all__ = ["ParamType", "ParamDict", "printParams"]

################################################################################
# what a parameter cane
ParamType = Union[int, float, str]

################################################################################
# instances of a set of actualised parameters
ParamDict = Dict[str, ParamType]


################################################################################
def printParams(params: ParamDict | None) -> None:
    if params:
        print(f"Params...")
        for k, v in params.items():
            print(f"   {k} : {v}")


################################################################################
def bakeParams(params: ParamDict, paramSpecs: ParamDict) -> ParamDict:
    """For a given set of parms, and their defaults, bake them into
    a final set of params.

    :param: params - the set of params to pull from
    :paramSpecs: the param definitions and their defaults
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
