from typing import (
    Any,
    NoReturn,
    Optional,
)

from theseus.validators._validator import Validator


class Integer(Validator):
    def __init__(
        self,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> None:
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError('min_value must be less or equal than max_value')

        self._min_value = min_value
        self._max_value = max_value

    def validate(
        self,
        value: Any,
    ) -> None:
        if not isinstance(value, int):
            raise TypeError(f'expected value {value} to be an int')

        if self._min_value is not None and value < self._min_value:
            raise ValueError(f'expected value {value} to be greater or equal than {self._min_value}')

        if self._max_value is not None and value > self._max_value:
            raise ValueError(f'expected value {value} to be less or equal than {self._max_value}')
