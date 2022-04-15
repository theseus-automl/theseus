import pytest

from tests.not_raises import not_raises
from tests.test_validators.setup_class import setup_class_with_validator
from theseus.validators.one_of import OneOf


@pytest.mark.parametrize(
    [
        'val',
        'expected',
        'exception',
    ],
    [
        (
            0,
            [1, 2, 3],
            ValueError,
        ),
        (
            0,
            [0, 1, 2],
            None,
        )
    ]
)
def test_one_of(
    val,
    expected,
    exception,
) -> None:
    ctx = not_raises if exception is None else pytest.raises
    dtype = setup_class_with_validator(
        OneOf,
        expected,
    )

    with ctx(exception):
        dtype(val)
