from typing import (
    Any,
    Type,
)

from theseus.validators._validator import Validator


def setup_class_with_validator(
    validator: Type[Validator],
    *args,
    **kwargs,
) -> Type:
    class Dummy:
        val = validator(
            *args,
            **kwargs,
        )

        def __init__(
            self,
            val: Any,
        ) -> None:
            self.val = val

    return Dummy
