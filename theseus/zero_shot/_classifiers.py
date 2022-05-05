from types import MappingProxyType
from typing import (
    List,
    Union,
)

import numpy as np
from transformers import pipeline

from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode

_MULTILANG_MODEL = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
_SUPPORTED_LANGS = MappingProxyType({
    LanguageCode.ENGLISH: 'typeform/distilbert-base-uncased-mnli',
    LanguageCode.SPANISH: 'Recognai/bert-base-spanish-wwm-cased-xnli',
    LanguageCode.FRENCH: 'BaptisteDoyen/camembert-base-xnli',
    LanguageCode.GERMAN: 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli',
    LanguageCode.NORWEGIAN: 'NbAiLab/nb-bert-base-mnli',

    LanguageCode.GREEK: _MULTILANG_MODEL,
    LanguageCode.BULGARIAN: _MULTILANG_MODEL,
    LanguageCode.RUSSIAN: _MULTILANG_MODEL,
    LanguageCode.TURKISH: _MULTILANG_MODEL,
    LanguageCode.ARABIC: _MULTILANG_MODEL,
    LanguageCode.VIETNAMESE: _MULTILANG_MODEL,
    LanguageCode.THAI: _MULTILANG_MODEL,
    LanguageCode.CHINESE: _MULTILANG_MODEL,
    LanguageCode.HINDI: _MULTILANG_MODEL,
    LanguageCode.URDU: _MULTILANG_MODEL,
    LanguageCode.SVAHILI: _MULTILANG_MODEL,
})


class ZeroShotClassifier:
    def __init__(
        self,
        target_lang: LanguageCode,
        candidate_labels: List[str],
    ) -> None:
        if target_lang not in _SUPPORTED_LANGS:
            raise UnsupportedLanguageError(f'zero-shot classification is not available for {target_lang}')

        self._model = pipeline(
            'zero-shot-classification',
            model=_SUPPORTED_LANGS[target_lang],
            framework='pt',
        )
        self._candidate_labels = candidate_labels

    def __call__(
        self,
        texts: Union[str, List[str]],
    ) -> List[str]:
        if isinstance(texts, str):
            texts = [texts]

        return [entry['labels'][np.argmax(entry['scores'])] for entry in self._model(texts, self._candidate_labels)]

