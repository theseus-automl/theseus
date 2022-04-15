from typing import NoReturn

import pytest

from tests.not_raises import not_raises
from theseus.dataset.augmentations.back_translation import BackTranslationAugmenter
from theseus.dataset.augmentations.generation import (
    GPTAugmenter,
    GPTAugmenterShortInputWarning,
)
from theseus.dataset.augmentations.random_insertion import RandomInsertionAugmenter
from theseus.dataset.augmentations.random_replacement import RandomReplacementAugmenter


@pytest.fixture()
def setup_gpt_augmenter():
    return GPTAugmenter()


@pytest.fixture()
def setup_back_translation_augmenter():
    return BackTranslationAugmenter()


@pytest.fixture()
def setup_random_insertion_augmenter():
    return RandomInsertionAugmenter()


@pytest.fixture()
def setup_random_replacement_augmenter():
    return RandomReplacementAugmenter()


def test_gpt_augmenter_short_input(setup_gpt_augmenter) -> NoReturn:
    with pytest.warns(GPTAugmenterShortInputWarning):
        setup_gpt_augmenter('short input')


def test_gpt_augmenter(setup_gpt_augmenter) -> NoReturn:
    with not_raises(Exception):
        setup_gpt_augmenter('this is the normal input')


def test_back_translation_augmenter(setup_back_translation_augmenter) -> NoReturn:
    with not_raises(Exception):
        setup_back_translation_augmenter('this is the normal input')


def test_random_insertion_augmenter(setup_random_insertion_augmenter) -> NoReturn:
    with not_raises(Exception):
        setup_random_insertion_augmenter('this is the normal input')


def test_random_replacement_augmenter(setup_random_replacement_augmenter) -> NoReturn:
    with not_raises(Exception):
        setup_random_replacement_augmenter('this is the normal input')
