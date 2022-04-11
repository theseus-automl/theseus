from typing import Tuple

import numpy as np
import pandas as pd

from theseus.dataset._const import _STRATEGIES
from theseus.dataset.text_dataset import TextDataset


def check_balance(
    texts: pd.Series,
    labels: pd.Series,
) -> bool:
    df = pd.DataFrame(
        {
            'texts': texts,
            'labels': labels,
        },
    ).reset_index()
    lengths = set()

    for label in df['labels'].unique():
        lengths.add(len(df[df['labels'] == label]))

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
