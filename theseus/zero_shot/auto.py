from pathlib import Path
from typing import List

import pandas as pd

from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger
from theseus.plotting.classification import plot_class_distribution
from theseus.zero_shot._classifiers import ZeroShotClassifier

_logger = setup_logger(__name__)


class AutoZeroShotClassifier:
    def __init__(
        self,
        target_lang: LanguageCode,
        candidate_labels: List[str],
        out_path: Path,
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
        texts: List[str],
    ) -> None:
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
                texts,
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
