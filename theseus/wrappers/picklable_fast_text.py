from pathlib import Path
from typing import (
    Any,
    Dict,
)

import numpy as np
from fasttext import (
    FastText,
    load_model,
)

FastText.eprint = lambda _: None


class PicklableFastText:
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

    def get_sentence_vector(
        self,
        text: str,
    ) -> np.ndarray:
        return self._model.get_sentence_vector(text)

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
