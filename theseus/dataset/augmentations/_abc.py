from abc import (
    ABC,
    abstractmethod,
)
from typing import Mapping

import torch
from transformers import pipeline

from theseus._inference import gc_with_cuda
from theseus.dataset.augmentations._models import FILL_MASK_MODELS
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode


class AbstractAugmenter(ABC):
    def __init__(
        self,
        target_lang: LanguageCode,
        supported_models: Mapping[LanguageCode, str],
        task: str,
        device: torch.device,
    ) -> None:
        if target_lang not in supported_models:
            raise UnsupportedLanguageError(f'{target_lang} language is not supported')

        self._pipeline = pipeline(
            task,
            model=supported_models[target_lang],
            framework='pt',
            device_num=device.index,
        )

    @abstractmethod
    def __call__(
        self,
        text: str,
    ) -> str:
        raise NotImplementedError

    def free(
        self,
    ) -> None:
        self._pipeline.model.to('cpu')
        gc_with_cuda()


class FillMaskAugmenter(AbstractAugmenter):
    def __init__(
        self,
        target_lang: LanguageCode,
        device: torch.device,
    ) -> None:
        super().__init__(
            target_lang,
            FILL_MASK_MODELS,
            'fill-mask',
            device,
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
