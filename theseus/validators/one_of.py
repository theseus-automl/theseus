from typing import (
    Any,
    Container,
    NoReturn,
)

from theseus.validators._validator import Validator


class OneOf(Validator):
    def __init__(
        self,
        expected: Container[Any],
    ) -> NoReturn:
        self._expected = expected

    def validate(
        self,
        value: Any,
    ) -> NoReturn:
        if value not in self._expected:
            raise ValueError('unexpected value')
