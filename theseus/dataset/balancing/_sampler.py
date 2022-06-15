from abc import ABC
from typing import (
    Dict,
    Tuple,
)

import pandas as pd

from theseus.dataset._const import (
    _BALANCING_THRESHOLD,
    _STRATEGIES,
)
from theseus.log import setup_logger
from theseus.validators.one_of import OneOf

_logger = setup_logger(__name__)


def _prepare(
    texts: pd.Series,
    labels: pd.Series,
    strategy: str,
) -> Tuple[pd.DataFrame, Dict[int, int], int, int]:
    if strategy not in _STRATEGIES:
        raise ValueError(f'unknown strategy "{strategy}". Possible values are: {", ".join(_STRATEGIES.keys())}')

    _logger.info(f'using strategy {strategy}')

    df = pd.DataFrame(
        {
            'texts': texts,
            'labels': labels,
        },
    ).reset_index()
    counts = {label: len(df[df['labels'] == label]) for label in df['labels'].unique()}
    major_class_samples = _STRATEGIES[strategy](counts.values())
    target_samples = int(_BALANCING_THRESHOLD[strategy] * major_class_samples)

    _logger.info(f'major class has {major_class_samples} samples')
    _logger.info(f'other class will be sampled to have {target_samples} sampler')

    return (
        df,
        counts,
        target_samples,
        major_class_samples,
    )


class _Sampler(ABC):
    strategy = OneOf(set(_STRATEGIES.keys()))

    def __init__(
        self,
        strategy: str,
    ) -> None:
        self._strategy = strategy
