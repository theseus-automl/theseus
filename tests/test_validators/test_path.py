from pathlib import Path
from typing import (
    Any,
    NoReturn,
    Type,
)

import pytest

from tests.not_raises import not_raises
from theseus.validators import (
    ExistingDir,
    ExistingFile,
    ExistingPath,
)
from setup_class import setup_class_with_validator

_CWD = Path(__file__).parent
_TEST_DIR = _CWD / 'test_directory'
_TEST_FILE_NO_EXT = _CWD / 'test_file'
_TEST_FILE_TXT = _CWD / 'test_file.txt'
_TEST_FILE_BIN = _CWD / 'test_file.bin'


@pytest.fixture()
def setup_files_and_dirs(
    request,
) -> NoReturn:
    _TEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    _TEST_FILE_NO_EXT.touch(exist_ok=True)
    _TEST_FILE_TXT.touch(exist_ok=True)
    _TEST_FILE_BIN.touch(exist_ok=True)

    def finalize():
        _TEST_DIR.rmdir()
        _TEST_FILE_NO_EXT.unlink(missing_ok=True)
        _TEST_FILE_TXT.unlink(missing_ok=True)
        _TEST_FILE_BIN.unlink(missing_ok=True)

    request.addfinalizer(finalize)


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
            _CWD / 'dummy',
            ValueError,
        ),
        (
            _TEST_FILE_TXT,
            ValueError,
        ),
        (
            _TEST_DIR,
            None,
        ),
    ],
)
def test_existing_dir_validator(
    setup_files_and_dirs,
    val: Any,
    exception: Type[Exception],
) -> NoReturn:
    ctx = not_raises if exception is None else pytest.raises
    dtype = setup_class_with_validator(ExistingDir)

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
            _CWD / 'dummy',
            ValueError,
        ),
        (
            _TEST_FILE_NO_EXT,
            ValueError,
        ),
        (
            _TEST_FILE_TXT,
            None,
        ),
        (
            _TEST_FILE_BIN,
            ValueError,
        ),
        (
            _TEST_DIR,
            ValueError,
        ),
    ],
)
def test_existing_file_validator_with_extension(
    setup_files_and_dirs,
    val: Any,
    exception: Type[Exception],
) -> NoReturn:
    ctx = not_raises if exception is None else pytest.raises
    dtype = setup_class_with_validator(
        ExistingFile,
        '.txt',
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
            _CWD / 'dummy',
            ValueError,
        ),
        (
            _TEST_FILE_NO_EXT,
            None,
        ),
        (
            _TEST_FILE_TXT,
            None,
        ),
        (
            _TEST_DIR,
            ValueError,
        ),
    ],
)
def test_existing_file_validator_no_extension(
    setup_files_and_dirs,
    val: Any,
    exception: Type[Exception],
) -> NoReturn:
    ctx = not_raises if exception is None else pytest.raises
    dtype = setup_class_with_validator(ExistingFile)

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
            _CWD / 'dummy',
            ValueError,
        ),
        (
            _TEST_FILE_TXT,
            None,
        ),
        (
            _TEST_DIR,
            None,
        ),
    ],
)
def test_existing_path_validator(
    setup_files_and_dirs,
    val: Any,
    exception: Type[Exception],
) -> NoReturn:
    ctx = not_raises if exception is None else pytest.raises
    dtype = setup_class_with_validator(ExistingPath)

    with ctx(exception):
        dtype(val)
