from abc import (
    ABC,
    abstractmethod,
)

import numpy as np
import pandas as pd

from theseus.dataset.balancing._sampler import (
    _prepare,
    _Sampler,
)
from theseus.dataset.text_dataset import TextDataset


class _RandomSampler(_Sampler, ABC):
    def __call__(
        self,
        dataset: TextDataset,
    ) -> TextDataset:
        df, counts, target_samples = _prepare(
            dataset.texts,
            dataset.labels,
            self._strategy,
        )

        for label, n_samples in counts.items():
            if n_samples != target_samples:
                df = self._update(
                    df,
                    abs(n_samples - target_samples),
                    label,
                )

        return TextDataset(
            df['texts'],
            df['labels'],
        )

    @staticmethod
    @abstractmethod
    def _update(
        df: pd.DataFrame,
        n_samples: int,
        label: int,
    ) -> pd.DataFrame:
        raise NotImplementedError


class RandomUnderSampler(_RandomSampler):
    def __init__(
        self,
    ) -> None:
        super().__init__('under')

    @staticmethod
    def _update(
        df: pd.DataFrame,
        n_samples: int,
        label: int,
    ) -> pd.DataFrame:
        to_drop = np.random.choice(
            df[df['labels'] == label].index,
            n_samples,
            replace=False,
        )
        return df.drop(to_drop)


class RandomOverSampler(_RandomSampler):
    def __init__(
        self,
    ) -> None:
        super().__init__('over')

    @staticmethod
    def _update(
        df: pd.DataFrame,
        n_samples: int,
        label: int,
    ) -> pd.DataFrame:
        sampled = df[df['labels'] == label].sample(
            n=n_samples,
            replace=False,
        )
        return pd.concat(
            [
                df,
                sampled,
            ],
            ignore_index=True,
        )
