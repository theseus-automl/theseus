from logging import Logger
from typing import (
    List,
    Optional,
    Union, Dict,
)

from theseus.lang_code import LanguageCode
from theseus.lang_detection import LanguageDetector


class AutoEstimatorMixin:
    @staticmethod
    def _detect_lang(
        target_lang: Optional[LanguageCode],
        texts: Union[str, List[str]],
    ) -> LanguageCode:
        if target_lang is None:
            detector = LanguageDetector()

            return detector(texts)
        else:
            return target_lang

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
