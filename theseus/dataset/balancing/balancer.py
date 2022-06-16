from copy import deepcopy
from typing import Optional

import numpy as np
import torch
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
from theseus.log import setup_logger
from theseus.utils import get_args_names

_RANDOM_THRESHOLD = 0.05
_SIMILARITY_THRESHOLD = 0.1
_AUGMENTATIONS_THRESHOLD = 0.5

_logger = setup_logger(__name__)


class DatasetBalancer:
    def __init__(
        self,
        target_lang: LanguageCode,
        ignore_imbalance: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self._target_lang = target_lang
        self._ignore_imbalance = ignore_imbalance
        self._device = device

    def __call__(
        self,
        dataset: TextDataset,
    ) -> TextDataset:
        if dataset.labels is None:
            raise ValueError('unable to balance dataset without labels')

        is_balanced = check_balance(dataset)

        if is_balanced:
            _logger.warning('dataset is already balanced. Performing no actions')

            return dataset

        if self._ignore_imbalance:
            _logger.warning('dataset is imbalanced, but param ignore_imbalance was set to True. Performing no actions')

            return dataset

        under_dev, over_dev = get_abs_deviation(dataset)
        sampler = self._choose_sampler(
            under_dev,
            over_dev,
        )

        if sampler is not None:
            _logger.info(f'using {type(sampler).__name__} for balancing')
            dataset = sampler(dataset)

        dataset = deepcopy(dataset)
        classes = np.unique(dataset.labels)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=dataset.labels,
        )
        dataset.class_weights = {}

        for cls, weight in zip(classes, weights):
            dataset.class_weights[cls] = weight

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
            if 'device' in get_args_names(sampler_cls.__init__) and self._device is not None:
                kwargs = {'device': self._device}
            else:
                kwargs = {}

            try:
                return sampler_cls(
                    self._target_lang,
                    **kwargs,
                )
            except UnsupportedLanguageError:
                _logger.error(
                    f'{sampler_cls} is unavailable for language {self._target_lang}, so the class weight will be used',
                )
