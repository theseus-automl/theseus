from abc import ABC
from logging import Logger
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

from pytorch_lightning import seed_everything

from theseus import Accelerator
from theseus.defaults import RANDOM_STATE
from theseus.lang_code import LanguageCode
from theseus.lang_detection import LanguageDetector
from theseus.log import setup_logger

_logger = setup_logger(__name__)


class AutoEstimator(ABC):
    def __init__(
        self,
        out_dir: Path,
        accelerator: Accelerator,
        target_lang: Optional[LanguageCode] = None,
        *,
        use_tf_idf: bool,
        use_fasttext: bool,
        use_bert: bool,
        tf_idf_n_jobs: int,
        fast_text_n_jobs: int,
        tf_idf_n_iter: int,
        fast_text_n_iter: int,
    ) -> None:
        seed_everything(RANDOM_STATE)

        self._out_dir = out_dir
        self._accelerator = accelerator
        self._target_lang = target_lang
        self._use_tf_idf = use_tf_idf
        self._use_fasttext = use_fasttext
        self._use_bert = use_bert
        self._tf_idf_n_jobs = tf_idf_n_jobs
        self._fast_text_n_jobs = fast_text_n_jobs
        self._tf_idf_n_iter = tf_idf_n_iter
        self._fast_text_n_iter = fast_text_n_iter

        self._out_dir.mkdir(
            exist_ok=True,
            parents=True,
        )

    def _detect_lang(
        self,
        texts: Union[str, List[str]],
    ) -> None:
        if self._target_lang is None:
            detector = LanguageDetector()

            self._target_lang = detector(texts)
            _logger.debug(f'detect language is {self._target_lang}')
        else:
            _logger.debug(f'language was set by user to {self._target_lang}')

    @staticmethod
    def _log_score(
        logger: Logger,
        score: float,
        metrics: Dict[str, float],
        algorithm: str,
        refit: str,
    ) -> None:
        logger.info(f'best {refit} score with {algorithm}: {score:.4f}')

        for name, metric in metrics.items():
            logger.info(f'{name:<15} = {metric:.4f}')
