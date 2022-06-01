from pathlib import Path
from typing import (
    List,
    Optional,
)

import pandas as pd

from theseus.abc.auto_estimator import AutoEstimator
from theseus.accelerator import Accelerator
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger
from theseus.plotting.classification import plot_class_distribution
from theseus.zero_shot._classifiers import ZeroShotClassifier

_logger = setup_logger(__name__)


class AutoZeroShotClassifier(AutoEstimator):
    def __init__(
        self,
        out_dir: Path,
        accelerator: Accelerator,
        candidate_labels: List[str],
        target_lang: Optional[LanguageCode] = None,
    ) -> None:
        super().__init__(
            out_dir,
            accelerator,
            target_lang,
        )

        self._candidate_labels = candidate_labels

    def fit(
        self,
        dataset: TextDataset,
    ) -> None:
        _logger.info('detecting_language')
        self._detect_lang(dataset.texts)

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
                self._out_dir,
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
