from typing import NoReturn

import pandas as pd

from theseus.dataset.augmentations.back_translation import BackTranslationAugmenter
from theseus.dataset.augmentations.gpt import GPTAugmenter
from theseus.dataset.augmentations.random_insertion import RandomInsertionAugmenter
from theseus.dataset.augmentations.random_replacement import RandomReplacementAugmenter
from theseus.dataset.balancing._sampler import _prepare


class AugmentationOverSampler:
    def __init__(
        self,
        target_lang: str,
    ) -> NoReturn:
        self._target_lang = target_lang
        self._augmenters = []

    def __call__(
        self,
        texts: pd.Series,
        labels: pd.Series,
    ) -> pd.DataFrame:
        df, counts, target_samples = _prepare(
            texts,
            labels,
            'over',
        )

        for label, n_samples in counts.items():
            if n_samples != target_samples:
                base = df[df['labels'] == label].sample(
                    n=abs(n_samples - target_samples),
                    replace=False,
                )['texts'].tolist()
                augmented = []

                for idx, text in enumerate(base):
                    augmented.append(self._augmenters[idx % len(self._augmenters)](text))

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

        return df

    def _select_augmenters(
        self,
    ) -> None:
        if self._target_lang == 'en':
            self._augmenters = [
                GPTAugmenter(),
                BackTranslationAugmenter(),
            ]
        else:
            self._augmenters = [
                RandomInsertionAugmenter(),
                RandomReplacementAugmenter(),
            ]
