from collections import namedtuple
from typing import (
    Any,
    Generator,
    List,
    Tuple,
    Union,
)

from sklearn.model_selection import StratifiedShuffleSplit

from theseus.dataset.text_dataset import TextDataset

_NoSplit = List[Tuple[slice, slice]]
_StratifiedSplit = Generator[Tuple[Any, Any], Any, None]

_SizeRange = namedtuple(
    '_SizeRange',
    'min max',
)

# everything else is considered as large
SMALL_DATASET_SIZE_RANGE = _SizeRange(
    min=50,
    max=9999,
)
MEDIUM_DATASET_SIZE_RANGE = _SizeRange(
    min=10000,
    max=99999,
)


def make_split(
    dataset: TextDataset,
) -> Union[_NoSplit, _StratifiedSplit]:
    if dataset.labels is None:
        return [(
            slice(None),
            slice(None),
        )]

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=select_test_size(len(dataset)),
    )

    return splitter.split(
        dataset.texts,
        dataset.labels,
    )


def select_test_size(
    dataset_size: int,
) -> float:
    if SMALL_DATASET_SIZE_RANGE.min <= dataset_size <= SMALL_DATASET_SIZE_RANGE.max:
        return 0.3

    if MEDIUM_DATASET_SIZE_RANGE.min <= dataset_size <= MEDIUM_DATASET_SIZE_RANGE.max:
        return 0.2

    return 0.1
