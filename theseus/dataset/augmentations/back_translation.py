from typing import NoReturn

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from theseus.dataset.augmentations._abc import AbstractAugmenter
from theseus.dataset.augmentations._models import BACK_TRANSLATION_MODELS
from theseus.lang_code import LanguageCode


class BackTranslationAugmenter(AbstractAugmenter):
    def __init__(
        self,
        target_lang: LanguageCode,
    ) -> NoReturn:
        super().__init__(
            target_lang,
            BACK_TRANSLATION_MODELS,
            'translation_en_to_de',
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            'google/bert2bert_L-24_wmt_de_en',
            pad_token='<pad>',
            eos_token='</s>',
            bos_token='<s>',
        )
        self._model_de_to_en = AutoModelForSeq2SeqLM.from_pretrained('google/bert2bert_L-24_wmt_de_en')

    def __call__(
        self,
        text: str,
    ) -> str:
        en_to_de_output = self._pipeline(text)
        translated_text = en_to_de_output[0]['translation_text']

        input_ids = self._tokenizer(
            translated_text,
            return_tensors='pt',
            add_special_tokens=False,
        ).input_ids
        output_ids = self._model_de_to_en.generate(input_ids)[0]
        augmented_text = self._tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        )

        return augmented_text
