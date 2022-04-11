from copy import deepcopy

from theseus.dataset.balancing.augmentation import AugmentationOverSampler
from theseus.dataset.balancing.checkers import (
    check_balance,
    get_abs_deviation,
)
from theseus.dataset.balancing.random import (
    RandomOverSampler,
    RandomUnderSampler,
)
from theseus.dataset.balancing.similarity import (
    SimilarityOverSampler,
    SimilarityUnderSampler,
)
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode

_RANDOM_THRESHOLD = 0.05
_SIMILARITY_THRESHOLD = 0.1
_AUGMENTATIONS_THRESHOLD = 0.5


class DatasetBalancer:
    def __init__(
        self,
        target_lang: LanguageCode,
        ignore_imbalance: bool = False,
        copy: bool = True,
    ) -> None:
        self._target_lang = target_lang
        self._ignore_imbalance = ignore_imbalance
        self._copy = copy

    def __call__(
        self,
        dataset: TextDataset,
    ) -> TextDataset:
        if dataset.labels is None:
            raise ValueError('unable to balance dataset without labels')

        if self._copy:
            dataset = deepcopy(dataset)

        is_balanced = check_balance(
            dataset.texts,
            dataset.labels,
        )

        if is_balanced:
            # todo: logging
            print('')
        else:
            if self._ignore_imbalance:
                # todo: logging
                print('')
            else:
                under_dev, over_dev = get_abs_deviation(dataset)
                sampler = self._choose_sampler(
                    under_dev,
                    over_dev,
                )

        return dataset

    def _choose_sampler(
        self,
        under_dev: float,
        over_dev: float,
    ):
        sampler_cls = None

        if under_dev <= _RANDOM_THRESHOLD:
            sampler_cls = RandomUnderSampler

        if over_dev <= _RANDOM_THRESHOLD:
            sampler_cls = RandomOverSampler

        if under_dev <= _SIMILARITY_THRESHOLD:
            sampler_cls = SimilarityUnderSampler

        if over_dev <= _SIMILARITY_THRESHOLD:
            sampler_cls = SimilarityOverSampler

        if over_dev <= _AUGMENTATIONS_THRESHOLD:
            sampler_cls = AugmentationOverSampler

        try:
            return sampler_cls(self._target_lang)
        except UnsupportedLanguageError:
            print('')  # TODO: logging

        return None
