from random import (
    choice,
    randint,
)
from string import ascii_letters

import pandas as pd

from theseus.dataset.text_dataset import TextDataset
from theseus.utils import chunkify

_MIN_TEXT_LEN = 5
_MAX_TEXT_LEN = 20
_MIN_SPLITS = 4
_MAX_SPLITS = _MAX_TEXT_LEN
_DATASET_SIZE = 100


def random_split(
    text: str,
) -> str:
    max_chunks = min(
        len(text) - 1,
        _MAX_SPLITS,
    )
    return ' '.join(''.join(chunk) for chunk in chunkify(list(text), randint(_MIN_SPLITS, max_chunks)))


def generate_text() -> str:
    text = ''.join(choice(ascii_letters) for _ in range(randint(_MIN_TEXT_LEN, _MAX_TEXT_LEN)))
    text = random_split(text)

    return text


def prepare_balanced_dataset() -> TextDataset:
    return TextDataset(
        pd.Series([generate_text() for _ in range(_DATASET_SIZE)]),
        pd.Series([0 for _ in range(_DATASET_SIZE // 2)] + [1 for _ in range(_DATASET_SIZE // 2)]),
    )


def prepare_imbalanced_dataset() -> TextDataset:
    return TextDataset(
        pd.Series([generate_text() for _ in range(_DATASET_SIZE)]),
        pd.Series(
            [0 for _ in range(_DATASET_SIZE // 2)] +
            [1 for _ in range(_DATASET_SIZE // 4)] +
            [2 for _ in range(_DATASET_SIZE // 4)]
        ),
    )
