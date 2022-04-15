from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    NoReturn,
    Optional,
    Type,
)


class Validator(ABC):
    def __get__(
        self,
        obj: Any,
        dtype: Optional[Type] = None,
    ) -> Any:
        return self.value

    def __set__(
        self,
        obj: Any,
        value: Any,
    ) -> None:
        self.validate(value)
        self.value = value

    @abstractmethod
    def validate(
        self,
        value: Any,
    ) -> None:
        pass
