import pytest

from tests.not_raises import not_raises
from theseus.dataset.augmentations.back_translation import BackTranslationAugmenter
from theseus.dataset.augmentations.generation import (
    GPTAugmenter,
    GPTAugmenterShortInputWarning,
)
from theseus.dataset.augmentations.random import (
    RandomInsertionAugmenter,
    RandomReplacementAugmenter,
)
from theseus.dataset.augmentations._models import (
    BACK_TRANSLATION_MODELS,
    FILL_MASK_MODELS,
    GENERATION_MODELS,
)

_TARGET_LANG = list(
    set(BACK_TRANSLATION_MODELS.keys()) & set(FILL_MASK_MODELS.keys()) & set(GENERATION_MODELS.keys()),
)[0]


@pytest.fixture()
def setup_gpt_augmenter():
    return GPTAugmenter(_TARGET_LANG)


@pytest.fixture()
def setup_back_translation_augmenter():
    return BackTranslationAugmenter(_TARGET_LANG)


@pytest.fixture()
def setup_random_insertion_augmenter():
    return RandomInsertionAugmenter(_TARGET_LANG)


@pytest.fixture()
def setup_random_replacement_augmenter():
    return RandomReplacementAugmenter(_TARGET_LANG)


def test_gpt_augmenter_short_input(setup_gpt_augmenter) -> None:
    with pytest.warns(GPTAugmenterShortInputWarning):
        setup_gpt_augmenter('short input')


def test_gpt_augmenter(setup_gpt_augmenter) -> None:
    with not_raises(Exception):
        setup_gpt_augmenter('this is the normal input')


def test_back_translation_augmenter(setup_back_translation_augmenter) -> None:
    with not_raises(Exception):
        setup_back_translation_augmenter('this is the normal input')


def test_random_insertion_augmenter(setup_random_insertion_augmenter) -> None:
    with not_raises(Exception):
        setup_random_insertion_augmenter('this is the normal input')


def test_random_replacement_augmenter(setup_random_replacement_augmenter) -> None:
    with not_raises(Exception):
        setup_random_replacement_augmenter('this is the normal input')
