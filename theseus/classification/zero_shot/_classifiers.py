from typing import (
    List,
    NoReturn,
    Union,
)

import numpy as np
from transformers import pipeline

from theseus._const import LanguageCode


_MONOLINGUAL_MODELS = {
    LanguageCode.english: 'facebook/bart-large-mnli',
}


class ZeroShotClassifier:
    def __init__(
        self,
        model_name: str,
        candidate_labels: List[str],
    ) -> NoReturn:
        self._model_name = model_name
        self._model = pipeline(
            'zero-shot-classification',
            model=model_name,
        )
        self._candidate_labels = candidate_labels

    def __call__(
        self,
        texts: Union[str, List[str]],
    ) -> List[str]:
        if isinstance(texts, str):
            texts = [texts]

        res = [entry['labels'][np.argmax(entry['scores'])] for entry in self._model(texts, self._candidate_labels)]

        return res

    @property
    def model_name(
        self,
    ) -> str:
        return self._model_name


class MultilingualZeroShotClassifier(ZeroShotClassifier):
    def __init__(
        self,
        candidate_labels: List[str],
    ) -> NoReturn:
        super().__init__(
            'joeddav/xlm-roberta-large-xnli',
            candidate_labels,
        )


class MonolingualZeroShotClassifier(ZeroShotClassifier):
    def __init__(
        self,
        lang_code: str,
        candidate_labels: List[str],
    ) -> NoReturn:
        if lang_code not in LanguageCode:
            raise ValueError(f'unknown language code {lang_code}')

        super().__init__(
            _MONOLINGUAL_MODELS[LanguageCode(lang_code)],
            candidate_labels,
        )
