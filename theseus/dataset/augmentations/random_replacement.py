from random import randint
from typing import NoReturn

from transformers import pipeline


class RandomReplacementAugmenter:
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

        idx = randint(
            1,
            len(tokens) - 1,
        )
        orig_word = tokens[idx]
        tokens[idx] = '[MASK]'
        tokens = ' '.join(tokens)
        augmentations = self._unmasker(tokens)

        for res in augmentations:
            if res['token_str'] != orig_word:
                augmented_text = res['sequence'].replace('[CLS] ', '').replace(' [SEP]', '')
                break
        else:
            raise ValueError('unable to generate output different from input')

        return augmented_text
