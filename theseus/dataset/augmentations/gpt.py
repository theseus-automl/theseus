import warnings
from random import randint

from transformers import pipeline

from theseus.validators import Integer


class GPTAugmenterShortInputWarning(Warning):
    pass


class GPTAugmenter:
    _min_input_len = Integer(min_value=5)
    _max_sequences = Integer(min_value=1)

    def __init__(
        self,
        min_input_len: int = 5,
        max_sequences: int = 10,
    ) -> NoReturn:
        self._min_input_len = min_input_len
        self._max_sequences = max_sequences

        self._generator = pipeline(
            'text-generation',
            model='gpt2',
        )

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
        output = self._generator(
            text,
            max_length=input_length + num_new_words,
            num_return_sequences=self._max_sequences,
        )
        augmented_text = output[randint(0, self._max_sequences - 1)]['generated_text']

        return augmented_text
