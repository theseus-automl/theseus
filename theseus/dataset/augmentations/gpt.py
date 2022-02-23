import warnings
from random import randint

from transformers import pipeline


class GPTAugmenterShortInputWarning(Warning):
    pass


class GPTAugmenter:
    def __init__(
        self,
        min_input_len: int = 5,
        max_sequences: int = 10,
    ) -> None:
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
                '',
                GPTAugmenterShortInputWarning,
            )

        num_new_words = randint(1, input_length)
        output = self._generator(
            text,
            max_length=input_length + num_new_words,
            num_return_sequences=self._max_sequences,
        )
        augmented_text = output[randint(0, self._max_sequences - 1)]['generated_text']

        return augmented_text
