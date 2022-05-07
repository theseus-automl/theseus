from typing import (
    List,
    Optional,
    Union,
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
