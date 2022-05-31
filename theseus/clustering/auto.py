from pathlib import Path
from typing import Optional

from pytorch_lightning import seed_everything

from theseus._mixin import AutoEstimatorMixin
from theseus.accelerator import Accelerator
from theseus.clustering.embeddings import (
    FastTextClusterer,
    SentenceBertClusterer,
    TfIdfClusterer,
)
from theseus.dataset.text_dataset import TextDataset
from theseus.defaults import RANDOM_STATE
from theseus.exceptions import DeviceError
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger

_logger = setup_logger(__name__)


class AutoClusterer(AutoEstimatorMixin):
    def __init__(
        self,
        out_dir: Path,
        accelerator: Accelerator,
        target_lang: Optional[LanguageCode] = None,
    ) -> None:
        seed_everything(RANDOM_STATE)

        self._out_dir = out_dir
        self._accelerator = accelerator
        self._target_lang = target_lang

        self._out_dir.mkdir(
            exist_ok=True,
            parents=True,
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

        # tfidf
        _logger.info('trying TF-IDF clustering')
        tf_idf_path = self._out_dir / 'tf-idf'
        tf_idf_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        clf = TfIdfClusterer(
            self._target_lang,
            tf_idf_path,
        )
        score, metrics = clf.fit(dataset)
        self._log_score(
            _logger,
            score,
            metrics,
            'TF-IDF',
            'silhouette',
        )

        # fasttext
        _logger.info('trying fastText clustering')
        ft_path = self._out_dir / 'ft'
        ft_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        clf = FastTextClusterer(
            self._target_lang,
            ft_path,
        )
        score, metrics = clf.fit(dataset)
        self._log_score(
            _logger,
            score,
            metrics,
            'fastText embeddings',
            'silhouette',
        )

        try:
            device = self._accelerator.select_single_gpu()
        except DeviceError:
            _logger.error('no suitable GPU was found, skipping SentenceBERT')
        else:
            # sentence_bert
            _logger.info('trying SentenceBERT clustering')
            sbert_path = self._out_dir / 'sbert'
            sbert_path.mkdir(
                parents=True,
                exist_ok=True,
            )
            clf = SentenceBertClusterer(
                self._target_lang,
                sbert_path,
                device,
            )
            score, metrics = clf.fit(dataset)
            self._log_score(
                _logger,
                score,
                metrics,
                'SentenceBERT embeddings',
                'silhouette',
            )
