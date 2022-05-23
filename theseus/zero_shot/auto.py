from pathlib import Path
from typing import (
    List,
    Optional,
)

import pandas as pd

from theseus import TextDataset
from theseus._mixin import AutoEstimatorMixin
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger
from theseus.plotting.classification import plot_class_distribution
from theseus.zero_shot._classifiers import ZeroShotClassifier

_logger = setup_logger(__name__)


class AutoZeroShotClassifier(AutoEstimatorMixin):
    def __init__(
        self,
        candidate_labels: List[str],
        out_path: Path,
        target_lang: Optional[LanguageCode] = None,
    ) -> None:
        self._target_lang = target_lang
        self._candidate_labels = candidate_labels
        self._out_path = out_path
        self._out_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    def fit(
        self,
        dataset: TextDataset,
    ) -> None:
        _logger.info('detecting_language')
        self._target_lang = self._detect_lang(
            self._target_lang,
            dataset.texts,
        )

        try:
            clf = ZeroShotClassifier(
                self._target_lang,
                self._candidate_labels,
            )
        except UnsupportedLanguageError as err:
            _logger.error(str(err))
        else:
            self._fit_single_model(
                clf,
                dataset.texts,
                self._out_path,
            )

    @staticmethod
    def _fit_single_model(
        model: ZeroShotClassifier,
        texts: List[str],
        out_path: Path,
    ) -> None:
        df = pd.DataFrame()
        df['texts'] = texts
        df['labels'] = [model(text) for text in texts]

        df_path = out_path / 'predictions.parquet.gzip'
        _logger.info(f'saving predictions to {df_path}')
        df.to_parquet(
            df_path,
            compression='gzip',
        )

        plot_class_distribution(
            df['labels'],
            out_path / 'class_distribution.png',
        )
