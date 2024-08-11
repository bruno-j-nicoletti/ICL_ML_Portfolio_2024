from .hopper import *
from .halfCheetah import *
from .invertedPendulum import *
from .physicalModel import *
from .walker import *

__all__ = ["fetchPhysicalModelSpec"]


def fetchPhysicalModelSpec(
        modelEnum: PhysicalModelID | str) -> PhysicalModelSpec:
    """Fetch the named physical model."""
    if modelEnum == PhysicalModelID.invertedPendulum:
        return InvertedPendulumSpec()
    elif modelEnum == PhysicalModelID.hopper:
        return HopperSpec()
    elif modelEnum == PhysicalModelID.halfCheetah:
        return HalfCheetahSpec()
    elif modelEnum == PhysicalModelID.walker:
        return WalkerSpec()
    raise Exception(f"Unknown model ({modelEnum})")
