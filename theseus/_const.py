from enum import (
    Enum,
    EnumMeta,
)
from typing import Any


class ContainsEnumMeta(EnumMeta):
    def __contains__(
        cls,
        item: Any,
    ) -> bool:
        return item in [v.value for v in cls.__members__.values()]


class LanguageCode(str, Enum, metaclass=ContainsEnumMeta):
    english = 'en'
