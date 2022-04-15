import warnings
from random import randint

from theseus.dataset.augmentations._abc import AbstractAugmenter
from theseus.dataset.augmentations._models import GENERATION_MODELS
from theseus.lang_code import LanguageCode
from theseus.validators import Integer


class GPTAugmenterShortInputWarning(Warning):
    pass


class GPTAugmenter(AbstractAugmenter):
    _min_input_len = Integer(min_value=5)
    _max_sequences = Integer(min_value=1)

    def __init__(
        self,
        target_lang: LanguageCode,
        min_input_len: int = 5,
        max_sequences: int = 10,
    ) -> None:
        super().__init__(
            target_lang,
            GENERATION_MODELS,
            'text-generation',
        )

        self._min_input_len = min_input_len
        self._max_sequences = max_sequences

    def __call__(
        self,
        text: str,
    ) -> str:
        input_length = len(text.split())

        if input_length < self._min_input_len:
            warnings.warn(
                'Input is too short. Results may be inaccurate',
                GPTAugmenterShortInputWarning,
            )

        num_new_words = randint(input_length // 2, input_length)
        output = self._pipeline(
            text,
            max_length=input_length + num_new_words,
            num_return_sequences=self._max_sequences,
        )

        return output[randint(0, self._max_sequences - 1)]['generated_text']
