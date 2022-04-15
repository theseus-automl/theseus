from itertools import chain

import pytest

from theseus.utils import (
    chunkify,
    extract_kwargs,
)


def test_invalid_num_chunks() -> None:
    with pytest.raises(ValueError):
        list(
            chunkify(
                [],
                0,
            ),
        )


def test_chunkify_empty_array() -> None:
    with pytest.raises(ValueError):
        list(
            chunkify(
                [],
                10,
            ),
        )


def test_num_chunks_gt_array_len() -> None:
    length = 5
    arr = list(range(length))

    assert list(chunkify(arr, length + 1))[0][0] == arr


def test_num_chunks_eq_array_len() -> None:
    length = 5
    arr = list(range(length))

    assert list(chunkify(arr, length)) == [[elem] for elem in arr]


@pytest.mark.parametrize(
    (
        'length',
        'num_chunks',
    ),
    (
        (
            5,
            2,
        ),
        (
            20,
            4,
        )
    )
)
def test_num_chunks_lt_array_len(
    length: int,
    num_chunks: int,
) -> None:
    arr = list(range(length))
    chunks = list(chunkify(arr, num_chunks))
    sizes = [len(chunk) for chunk in chunks]

    pytest.assume(set(arr) == set(chain.from_iterable(chunks)))
    pytest.assume(sizes[i] >= sizes[i + 1] for i in range(len(sizes) - 1))


def test_extract_kwargs_no_args() -> None:
    def dummy():
        pass

    assert extract_kwargs(dummy, a=1) == {}


def test_extract_kwargs_no_matching_args() -> None:
    def dummy(
        a,
        b,
    ):
        pass

    assert extract_kwargs(dummy, c=1, d=2) == {}


def test_extract_kwargs_all_matching_args() -> None:
    def dummy(
        a,
        b,
    ):
        pass

    assert extract_kwargs(dummy, a=1, b=2) == {'a': 1, 'b': 2}


def test_extract_kwargs_not_all_matching_args() -> None:
    def dummy(
        a,
        b,
        c,
    ):
        pass

    assert extract_kwargs(dummy, a=1, b=2, d=3) == {'a': 1, 'b': 2}
