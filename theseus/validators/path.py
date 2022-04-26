import re
from pathlib import Path
from typing import (
    Any,
    Optional,
)

from theseus.validators._validator import Validator

_EXTENSION_RE = re.compile(r'^\.\w+$')


class ExistingPath(Validator):
    def validate(
        self,
        value: Any,
    ) -> None:
        if not isinstance(value, Path):
            raise TypeError(f'expected value {value} to be a Path')

        if not value.exists():
            raise ValueError(f'path {value} does not exist')


class ExistingFile(ExistingPath):
    def __init__(
        self,
        extension: Optional[str] = None,
    ) -> None:
        if extension is not None and not _EXTENSION_RE.match(extension):
            raise ValueError('invalid file extension')

        self._extension = extension

    def validate(
        self,
        value: Any,
    ) -> None:
        super().validate(value)

        if not value.is_file():
            raise ValueError(f'file {value} does not exist')

        if self._extension is not None and value.suffix != self._extension:
            raise ValueError(f'expected file {value} to have extension {self._extension}, but found {value.suffix}')


class ExistingDir(ExistingPath):
    def validate(
        self,
        value: Any,
    ) -> None:
        super().validate(value)

        if not value.is_dir():
            raise ValueError(f'directory {value} does not exist')
