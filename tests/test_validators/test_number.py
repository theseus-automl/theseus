from typing import (
    Any,
    NoReturn,
    Type,
)

import pytest

from tests.not_raises import not_raises
from theseus.validators import Integer
from setup_class import setup_class_with_validator


def test_invalid_restrictions() -> NoReturn:
    with pytest.raises(ValueError):
        class Dummy:
            val = Integer(
                1000,
                100,
            )


@pytest.mark.parametrize(
    [
        'val',
        'exception',
    ],
    [
        (
            'val',
            TypeError,
        ),
        (
            100,
            None,
        ),
    ],
)
def test_no_restrictions(
    val: Any,
    exception: Type[Exception],
) -> NoReturn:
    ctx = not_raises if exception is None else pytest.raises
    dtype = setup_class_with_validator(Integer)

    with ctx(exception):
        dtype(val)


@pytest.mark.parametrize(
    [
        'val',
        'exception',
    ],
    [
        (
            'val',
            TypeError,
        ),
        (
            0,
            ValueError,
        ),
        (
            15,
            None,
        ),
    ],
)
def test_only_min_value(
    val: Any,
    exception: Type[Exception],
) -> NoReturn:
    ctx = not_raises if exception is None else pytest.raises
    dtype = setup_class_with_validator(
        Integer,
        min_value=10,
    )

    with ctx(exception):
        dtype(val)


@pytest.mark.parametrize(
    [
        'val',
        'exception',
    ],
    [
        (
            'val',
            TypeError,
        ),
        (
            15,
            ValueError,
        ),
        (
            0,
            None,
        ),
    ],
)
def test_only_max_value(
    val: Any,
    exception: Type[Exception],
) -> NoReturn:
    ctx = not_raises if exception is None else pytest.raises
    dtype = setup_class_with_validator(
        Integer,
        max_value=10,
    )

    with ctx(exception):
        dtype(val)


@pytest.mark.parametrize(
    [
        'val',
        'exception',
    ],
    [
        (
            'val',
            TypeError,
        ),
        (
            0,
            ValueError,
        ),
        (
            15,
            ValueError,
        ),
        (
            7,
            None,
        ),
    ],
)
def test_with_restrictions(
    val: Any,
    exception: Type[Exception],
) -> NoReturn:
    ctx = not_raises if exception is None else pytest.raises
    dtype = setup_class_with_validator(
        Integer,
        min_value=5,
        max_value=10,
    )

    with ctx(exception):
        dtype(val)
