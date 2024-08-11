import numpy as np
from typing import Any, Dict, List, Sequence

__all__ = ["computeScore"]


def computeScore(
        totalRewards: Sequence[float] | np.ndarray) -> Dict[str, float]:
    """Computes a score over a set of model runs,

      :param: totalRewards : a list, one for each run, being the total of all rewards
                       for that run.
    """
    array = np.array(totalRewards)
    mean = array.mean()
    cov = array.std() / abs(mean)  # coefficient of variation
    score = mean * (1 - cov)
    if mean < 0 and score > 0:
        score = -score
    return ({"mean": mean, "cov": cov, "score": score})
