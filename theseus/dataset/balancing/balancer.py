from copy import deepcopy

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

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
from theseus.dataset.balancing.types import SamplerType
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
    ) -> None:
        self._target_lang = target_lang
        self._ignore_imbalance = ignore_imbalance

    def __call__(
        self,
        dataset: TextDataset,
    ) -> TextDataset:
        if dataset.labels is None:
            raise ValueError('unable to balance dataset without labels')

        is_balanced = check_balance(
            dataset.texts,
            dataset.labels,
        )

        if is_balanced:
            # todo: logging
            print('')
            return dataset

        if self._ignore_imbalance:
            # todo: logging
            print('')

            return dataset

        under_dev, over_dev = get_abs_deviation(dataset)
        sampler = self._choose_sampler(
            under_dev,
            over_dev,
        )

        if sampler is None:
            dataset = deepcopy(dataset)
            dataset.class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(dataset.labels),
                y=dataset.labels,
            )
        else:
            dataset = sampler(dataset)

        return dataset

    def _choose_sampler(
        self,
        under_dev: float,
        over_dev: float,
    ) -> SamplerType:
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

        if sampler_cls is not None:
            try:
                return sampler_cls(self._target_lang)
            except UnsupportedLanguageError:
                print('')  # TODO: logging
