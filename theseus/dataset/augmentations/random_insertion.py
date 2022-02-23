from random import randint
from typing import NoReturn

from transformers import pipeline


class RandomInsertionAugmenter:
    def __init__(
        self,
    ) -> NoReturn:
        self._unmasker = pipeline(
            'fill-mask',
            model='bert-base-multilingual-cased',
        )

    def __call__(
        self,
        text: str,
    ) -> str:
        tokens = text.split()

        idx = randint(1, len(tokens) - 2)
        masked = ' '.join(tokens[:idx] + ['[MASK]'] + tokens[idx:])
        augmented_text = self._unmasker(masked)[0]['sequence']

        return augmented_text
