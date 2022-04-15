import pytest

from tests.test_balancing.dataset import (
    prepare_balanced_dataset,
    prepare_imbalanced_dataset,
)
from theseus.dataset.balancing.augmentation import AugmentationOverSampler
from theseus.dataset.balancing.checkers import check_balance
from theseus.lang_code import LanguageCode


@pytest.fixture()
def setup_augmentation_over_sampler():
    return AugmentationOverSampler(LanguageCode.ENGLISH)


@pytest.mark.parametrize(
    'generator',
    (
        prepare_balanced_dataset,
        prepare_imbalanced_dataset,
    )
)
def test_augmentation_over_sampler(
    generator,
    setup_augmentation_over_sampler,
) -> None:
    dataset = setup_augmentation_over_sampler(generator())

    assert check_balance(dataset)
