from pathlib import Path
from typing import (
    List,
    Optional,
)

import pandas as pd

from theseus.abc.auto_estimator import AutoEstimator
from theseus.accelerator import Accelerator
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import (
    DeviceError,
    UnsupportedLanguageError,
)
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
        batch_size: int = 1,
        target_lang: Optional[LanguageCode] = None,
    ) -> None:
        super().__init__(
            out_dir,
            accelerator,
            target_lang,
            use_tf_idf=None,
            use_fasttext=None,
            use_bert=None,
            tf_idf_n_jobs=None,
            fast_text_n_jobs=None,
            tf_idf_n_iter=None,
            fast_text_n_iter=None,
        )

        self._batch_size = batch_size
        self._candidate_labels = candidate_labels

    def fit(
        self,
        dataset: TextDataset,
    ) -> None:
        _logger.info('detecting_language')
        self._detect_lang(dataset.texts)

        try:
            device_num = self._accelerator.select_single_gpu().index
        except DeviceError as err:
            _logger.error(f'DeviceError occurred: {err}')
            _logger.error('unable to select GPU, falling back to CPU...')
            device_num = -1
        else:
            _logger.debug(f'using GPU #{device_num}')

        try:
            clf = ZeroShotClassifier(
                self._target_lang,
                self._candidate_labels,
                device_num=device_num,
                batch_size=self._batch_size,
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
        df['text'] = texts
        df['label'], df['score'] = model(texts)  # noqa: WPS414

        df_path = out_path / 'predictions.parquet.gzip'
        _logger.info(f'saving predictions to {df_path}')
        df.to_parquet(
            df_path,
            compression='gzip',
        )

        plot_class_distribution(
            df['label'],
            out_path / 'class_distribution.png',
        )
