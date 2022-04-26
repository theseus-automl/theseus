from pathlib import Path
from typing import List

import pandas as pd

from theseus.plotting.classification import plot_class_distribution
from theseus.validators import ExistingDir
from theseus.zero_shot._classifiers import (
    MonolingualZeroShotClassifier,
    MultilingualZeroShotClassifier,
    ZeroShotClassifier,
)


class AutoZeroShotClassifier:
    _out_path = ExistingDir()

    def __init__(
        self,
        candidate_labels: List[str],
        lang: str,
        out_path: Path,
    ) -> None:
        self._candidate_labels = candidate_labels
        self._lang = lang
        self._out_path = out_path

    def fit(
        self,
        texts: List[str],
    ) -> None:
        multi = MultilingualZeroShotClassifier(self._candidate_labels)
        self._fit_single_model(
            multi,
            texts,
            self._out_path / f'{multi.model_name}',
        )

        try:
            mono = MonolingualZeroShotClassifier(
                self._lang,
                self._candidate_labels,
            )
        except ValueError:
            # TODO: logging
            pass
        else:
            self._fit_single_model(
                mono,
                texts,
                self._out_path / f'{mono.model_name}',
            )

    @staticmethod
    def _fit_single_model(
        model: ZeroShotClassifier,
        texts: List[str],
        out_path: Path,
    ) -> None:
        out_path.mkdir(
            exist_ok=True,
            parents=True,
        )

        df = pd.DataFrame()
        df['texts'] = texts
        df['labels'] = [model(text) for text in texts]

        df.to_parquet(
            out_path / 'predictions.parquet.gzip',
            compression='gzip',
        )

        plot_class_distribution(
            df['labels'],
            out_path / 'class_distribution.png',
        )
