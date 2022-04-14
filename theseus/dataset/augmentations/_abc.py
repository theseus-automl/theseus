from abc import (
    ABC,
    abstractmethod,
)
from types import MappingProxyType

from transformers import pipeline

from theseus.dataset.augmentations._bert_models import FILL_MASK_MODELS
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode


class AbstractAugmenter(ABC):
    def __init__(
        self,
        target_lang: LanguageCode,
        supported_models: MappingProxyType[LanguageCode, str],
        task: str,
    ) -> None:
        if target_lang not in supported_models:
            raise UnsupportedLanguageError(f'{target_lang} language is not supported')

        self._pipeline = pipeline(
            task,
            model=supported_models[target_lang],
        )

    @abstractmethod
    def __call__(
        self,
        text: str,
    ) -> str:
        raise NotImplementedError


class FillMaskAugmenter(AbstractAugmenter):
    def __init__(
        self,
        target_lang: LanguageCode,
    ) -> None:
        super().__init__(
            target_lang,
            FILL_MASK_MODELS,
            'fill-mask',
        )

        self._cls_token = self._pipeline.tokenizer.cls_token
        self._mask_token = self._pipeline.tokenizer.mask_token
        self._sep_token = self._pipeline.tokenizer.sep_token

    @abstractmethod
    def __call__(
        self,
        text: str,
    ) -> str:
        raise NotImplementedError
