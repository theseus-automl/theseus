from contextlib import contextmanager
from typing import Type


@contextmanager
def not_raises(
    expected_exception: Type[Exception],
) -> None:
    try:
        yield
    except expected_exception as error:
        raise AssertionError(f'raised exception {error} when it should not')
    except Exception as error:
        raise AssertionError(f'raised an unexpected exception {error}')
