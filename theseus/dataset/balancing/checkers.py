from typing import Tuple

import numpy as np

from theseus.dataset._const import _STRATEGIES
from theseus.dataset.text_dataset import TextDataset


def check_balance(
    dataset: TextDataset
) -> bool:
    lengths = set()
    _, counts = np.unique(
        dataset.labels,
        return_counts=True,
    )

    for count in counts:
        lengths.add(count)

    return len(lengths) == 1


def get_abs_deviation(
    dataset: TextDataset,
) -> Tuple[float, float]:
    values, counts = np.unique(
        dataset.labels,
        return_counts=True,
    )
    target_samples_under = _STRATEGIES['under'](counts)
    target_samples_over = _STRATEGIES['over'](counts)

    return (
        np.median([np.abs(val - target_samples_under) / target_samples_under for val in values]),
        np.median([np.abs(val - target_samples_over) / target_samples_over for val in values]),
    )
