import gc
from typing import NoReturn

import pandas as pd
import torch

from theseus.dataset.augmentations.back_translation import BackTranslationAugmenter
from theseus.dataset.augmentations.generation import GPTAugmenter
from theseus.dataset.augmentations.random_insertion import RandomInsertionAugmenter
from theseus.dataset.augmentations.random_replacement import RandomReplacementAugmenter
from theseus.dataset.balancing._sampler import _prepare
from theseus.dataset.text_dataset import TextDataset
from theseus.utils import chunkify


class AugmentationOverSampler:
    def __init__(
        self,
        target_lang: str,
    ) -> NoReturn:
        self._target_lang = target_lang
        self._augmenters = []

    def __call__(
        self,
        dataset: TextDataset,
    ) -> TextDataset:
        df, counts, target_samples = _prepare(
            dataset.texts,
            dataset.labels,
            'over',
        )

        for label, n_samples in counts.items():
            if n_samples != target_samples:
                base = df[df['labels'] == label].sample(
                    n=abs(n_samples - target_samples),
                    replace=False,
                )['texts'].tolist()
                augmented = []

                for model_cls, chunk in zip(self._augmenters, chunkify(base, len(self._augmenters))):
                    model = model_cls()

                    for text in chunk:
                        augmented.append(model(text))

                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

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
    ) -> None:
        if self._target_lang == 'en':
            self._augmenters = [
                GPTAugmenter,
                BackTranslationAugmenter,
            ]
        else:
            self._augmenters = [
                RandomInsertionAugmenter,
                RandomReplacementAugmenter,
            ]
