from abc import ABC
from typing import (
    Dict,
    NoReturn,
    Tuple,
)

import pandas as pd

from theseus.dataset._const import _STRATEGIES
from theseus.validators.one_of import OneOf


def _prepare(
    texts: pd.Series,
    labels: pd.Series,
    strategy: str,
) -> Tuple[pd.DataFrame, Dict[int, int], int]:
    if strategy not in _STRATEGIES:
        raise ValueError(
            f'unknown strategy "{strategy}". '
            f'Consider using one of the following: {", ".join(_STRATEGIES.keys())}',
        )

    df = pd.DataFrame(
        {
            'texts': texts,
            'labels': labels,
        },
    ).reset_index()
    counts = {label: len(df[df['labels'] == label]) for label in df['labels'].unique()}
    target_samples = _STRATEGIES[strategy](counts.values())

    return (
        df,
        counts,
        target_samples,
    )


class _Sampler(ABC):
    strategy = OneOf(set(_STRATEGIES.keys()))

    def __init__(
        self,
        strategy: str,
    ) -> None:
        self._strategy = strategy
