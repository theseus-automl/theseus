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


class AutoEstimator:
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

    def _detect_lang(
        self,
        texts: Union[str, List[str]],
    ) -> None:
        if self._target_lang is None:
            detector = LanguageDetector()

            self._target_lang = detector(texts)

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
