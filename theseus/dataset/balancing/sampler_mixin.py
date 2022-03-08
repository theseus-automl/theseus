from abc import ABC
from typing import (
    Dict,
    Tuple,
)

import pandas as pd

_STRATEGIES = {
    'under': min,
    'over': max,
}


class SamplerMixin(ABC):
    def __new__(
        cls,
        *args,
        **kwargs,
    ) -> 'SamplerMixin':
        if cls is SamplerMixin:
            raise TypeError(f'only children of "{cls.__name__}" may be instantiated')

        return object.__new__(
            cls,
            *args,
            **kwargs,
        )

    def __call__(
        self,
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
