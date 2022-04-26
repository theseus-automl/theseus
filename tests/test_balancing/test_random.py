import pytest

from tests.test_balancing.dataset import (
    prepare_balanced_dataset,
    prepare_imbalanced_dataset,
)
from theseus.dataset.balancing.checkers import check_balance
from theseus.dataset.balancing.random import (
    RandomOverSampler,
    RandomUnderSampler,
)


@pytest.fixture()
def setup_random_over_sampler() -> RandomOverSampler:
    return RandomOverSampler()


@pytest.fixture()
def setup_random_under_sampler() -> RandomUnderSampler:
    return RandomUnderSampler()


@pytest.mark.parametrize(
    'generator',
    (
        prepare_balanced_dataset,
        prepare_imbalanced_dataset,
    )
)
def test_random_over_sampler(
    generator,
    setup_random_over_sampler,
) -> None:
    dataset = setup_random_over_sampler(generator())

    assert check_balance(dataset)


@pytest.mark.parametrize(
    'generator',
    (
        prepare_balanced_dataset,
        prepare_imbalanced_dataset,
    )
)
def test_random_under_sampler(
    generator,
    setup_random_under_sampler,
) -> None:
    dataset = setup_random_under_sampler(generator())

    assert check_balance(dataset)
