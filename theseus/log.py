import logging
import sys
from enum import Enum


class _Color(Enum):
    green = '\u001b[32m'
    blue = '\u001b[34m'
    yellow = '\x1b[33;20m'
    red = '\x1b[31;20m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'


class ColoredFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            fmt,
            *args,
            **kwargs,
        )

        self._formats = {
            logging.DEBUG: logging.Formatter(_Color.green.value + fmt + _Color.reset.value),
            logging.INFO: logging.Formatter(_Color.blue.value + fmt + _Color.reset.value),
            logging.WARNING: logging.Formatter(_Color.yellow.value + fmt + _Color.reset.value),
            logging.ERROR: logging.Formatter(_Color.red.value + fmt + _Color.reset.value),
            logging.CRITICAL: logging.Formatter(_Color.bold_red.value + fmt + _Color.reset.value),
        }

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        return self._formats.get(record.levelno).format(record)


def setup_logger(
    name: str,
    level: int = logging.DEBUG,
    fmt: str = '%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s',
    **kwargs,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        ColoredFormatter(
            fmt,
            **kwargs,
        ),
    )
    logger.addHandler(stdout_handler)

    return logger
