from collections import Counter
from pathlib import Path
from typing import (
    List,
    Union,
)

from theseus.exceptions import ModelNotFoundError
from theseus.wrappers.picklable_fast_text import PicklableFastText

_FT_THRESHOLD = 0


class LanguageDetector:
    def __init__(
        self,
        model_path: Path,
    ) -> None:
        if not model_path.exists() or not model_path.is_file():
            raise ModelNotFoundError('language detection model does not exist or is not a file')

        self._model = PicklableFastText(model_path)

    def __call__(
        self,
        text: Union[str, List[str]],
    ) -> str:
        if isinstance(text, str):
            text = [text]

        predictions = Counter()

        for pred in self._model.predict(text, threshold=_FT_THRESHOLD, k=1)[0]:
            predictions.update(pred)

        return predictions.most_common(1)[0][0].replace('__label__', '')
