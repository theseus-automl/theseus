from collections import Counter
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Union,
)

from fasttext import (
    FastText,
    load_model,
)

from theseus.exceptions import ModelNotFoundError

_FT_THRESHOLD = 0

FastText.eprint = lambda _: None


class FasttextWrapper:
    def __init__(
        self,
        model_path: Path,
    ) -> None:
        self._model_path = model_path
        self._model = load_model(str(self._model_path))

    def predict(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        return self._model.predict(
            *args,
            **kwargs,
        )

    def __getstate__(
        self,
    ) -> Dict[str, Path]:
        return {'path': self._model_path}

    def __setstate__(
        self,
        state: Dict[str, Path],
    ) -> None:
        self._model_path = state['path']
        self._model = load_model(str(self._model_path))


class LanguageDetector:
    def __init__(
        self,
        model_path: Path,
    ) -> None:
        if not model_path.exists() or not model_path.is_file():
            raise ModelNotFoundError('language detection model does not exist or is not a file')

        self._model = FasttextWrapper(model_path)

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
