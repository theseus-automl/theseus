from typing import Callable

import pandas as pd
import pytest

from tests.test_samplers.dataset import (
    prepare_balanced_dataset,
    prepare_imbalanced_dataset,
)
from theseus.dataset.balancing._sampler import _prepare


def test_prepare_invalid_strategy() -> None:
    with pytest.raises(ValueError):
        _prepare(
            pd.Series(),
            pd.Series(),
            'dummy',
        )


@pytest.mark.parametrize(
    (
        'generator',
        'target_samples',
    ),
    (
        (
            prepare_balanced_dataset,
            50,
        ),
        (
            prepare_imbalanced_dataset,
            25,
        ),
    ),
)
def test_prepare_under_sampling(
    generator: Callable,
    target_samples: int,
) -> None:
    dataset = generator()
    _, _, calculated = _prepare(
        dataset.texts,
        dataset.labels,
        'under',
    )

    assert calculated == target_samples


@pytest.mark.parametrize(
    (
        'generator',
        'target_samples',
    ),
    (
        (
            prepare_balanced_dataset,
            50,
        ),
        (
            prepare_imbalanced_dataset,
            50,
        ),
    ),
)
def test_prepare_over_sampling(
    generator: Callable,
    target_samples: int,
) -> None:
    dataset = generator()
    _, _, calculated = _prepare(
        dataset.texts,
        dataset.labels,
        'over',
    )

    assert calculated == target_samples
