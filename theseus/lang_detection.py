from collections import Counter
from typing import (
    List,
    Union,
)
from urllib.request import urlretrieve

from theseus._paths import CACHE_DIR
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger
from theseus.wrappers.picklable_fast_text import PicklableFastText

_FT_THRESHOLD = 0

_logger = setup_logger(__name__)


class LanguageDetector:
    def __init__(
        self,
    ) -> None:
        self._model_path = CACHE_DIR / 'lid.176.bin'
        self._model = PicklableFastText(self._model_path)

    def __call__(
        self,
        text: Union[str, List[str]],
    ) -> LanguageCode:
        if isinstance(text, str):
            text = [text]

        predictions = Counter()

        for pred in self._model.predict(text, threshold=_FT_THRESHOLD, k=1)[0]:
            predictions.update(pred)

        return LanguageCode(predictions.most_common(1)[0][0].replace('__label__', ''))

    def _download_model(
        self,
    ) -> None:
        url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'

        if self._model_path.exists():
            _logger.debug('language identification model already exists, skipping download')
        else:
            _logger.debug(f'trying to download language identification model from {url}')

            self._model_path.touch(exist_ok=True)
            urlretrieve(
                url,
                self._model_path,
            )

        _logger.debug(f'language identification model is ready to use')
