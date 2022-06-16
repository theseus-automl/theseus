from typing import (
    List,
    Type,
)

import pandas as pd
import torch
from tqdm import tqdm

from theseus.dataset.augmentations._abc import AbstractAugmenter
from theseus.dataset.augmentations._models import (
    BACK_TRANSLATION_MODELS,
    FILL_MASK_MODELS,
    GENERATION_MODELS,
)
from theseus.dataset.augmentations.back_translation import BackTranslationAugmenter
from theseus.dataset.augmentations.generation import GPTAugmenter
from theseus.dataset.augmentations.random import (
    RandomInsertionAugmenter,
    RandomReplacementAugmenter,
)
from theseus.dataset.balancing._sampler import _prepare
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.utils import chunkify


class AugmentationOverSampler:
    def __init__(
        self,
        target_lang: LanguageCode,
        device: torch.device,
    ) -> None:
        self._target_lang = target_lang
        self._device = device
        self._augmenters = self._select_augmenters()

    def __call__(
        self,
        dataset: TextDataset,
    ) -> TextDataset:
        df, counts, target_samples, major_class_samples = _prepare(
            dataset.texts,
            dataset.labels,
            'over',
        )

        for label, n_samples in tqdm(counts.items()):
            if n_samples != major_class_samples:
                base = df[df['labels'] == label].sample(
                    n=abs(n_samples - target_samples),
                    replace=True,
                )['texts'].tolist()
                augmented = []

                for model_cls, chunk in tqdm(zip(self._augmenters, chunkify(base, len(self._augmenters)))):
                    model = model_cls(
                        target_lang=self._target_lang,
                        device=self._device,
                    )

                    for text in chunk:
                        augmented.append(model(text))

                    model.free()
                    del model

                df = pd.concat(
                    [
                        df,
                        pd.DataFrame({
                            'texts': augmented,
                            'labels': label,
                        }),
                    ],
                    ignore_index=True,
                )

        return TextDataset(
            df['texts'],
            df['labels'],
        )

    def _select_augmenters(
        self,
    ) -> List[Type[AbstractAugmenter]]:
        augmenters = []

        if self._target_lang in BACK_TRANSLATION_MODELS:
            augmenters.append(BackTranslationAugmenter)

        if self._target_lang in GENERATION_MODELS:
            augmenters.append(GPTAugmenter)

        if self._target_lang in FILL_MASK_MODELS:
            augmenters.append(RandomInsertionAugmenter)
            augmenters.append(RandomReplacementAugmenter)

        if not len(augmenters):
            raise UnsupportedLanguageError(f'none of the augmentations is available for {self._target_lang} language')

        return augmenters
