from random import (
    choice,
    randint,
)
from string import ascii_letters

import pandas as pd

from theseus.dataset.text_dataset import TextDataset

_MIN_TEXT_LEN = 5
_MAX_TEXT_LEN = 20
_DATASET_SIZE = 100


def generate_text() -> str:
    return ''.join(choice(ascii_letters) for _ in range(randint(_MIN_TEXT_LEN, _MAX_TEXT_LEN)))


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
