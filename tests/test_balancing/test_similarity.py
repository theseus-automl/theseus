import pytest

from tests.test_balancing.dataset import (
    prepare_balanced_dataset,
    prepare_imbalanced_dataset,
)
from theseus.dataset.balancing.checkers import check_balance
from theseus.dataset.balancing.similarity import (
    SimilarityOverSampler,
    SimilarityUnderSampler,
)
from theseus.lang_code import LanguageCode

_TARGET_LANG = LanguageCode.ENGLISH


@pytest.fixture()
def setup_similarity_under_sampler():
    return SimilarityUnderSampler(_TARGET_LANG)


@pytest.fixture()
def setup_similarity_over_sampler():
    return SimilarityOverSampler(_TARGET_LANG)


@pytest.mark.parametrize(
    'generator',
    (
        prepare_balanced_dataset,
        prepare_imbalanced_dataset,
    )
)
def test_random_over_sampler(
    generator,
    setup_similarity_over_sampler,
) -> None:
    dataset = setup_similarity_over_sampler(generator())

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
    setup_similarity_under_sampler,
) -> None:
    dataset = setup_similarity_under_sampler(generator())

    assert check_balance(dataset)
