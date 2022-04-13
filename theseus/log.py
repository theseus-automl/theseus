import logging
import sys
# from enum import Enum

from theseus.utils import extract_kwargs


# class _Color(Enum):
#     grey = '\x1b[38;20m'
#     yellow = '\x1b[33;20m'
#     red = '\x1b[31;20m'
#     bold_red = '\x1b[31;1m'
#     reset = '\x1b[0m'
#
#
# class ColoredFormatter(logging.Formatter):
#     def __init__(
#         self,
#         fmt: str,
#         *args,
#         **kwargs,
#     ) -> None:
#         super().__init__(
#             fmt,
#             *args,
#             **kwargs,
#         )
#
#         self._formats = {
#             logging.DEBUG: _Color.grey.value + fmt + _Color.reset.value,
#             logging.INFO: _Color.grey.value + fmt + _Color.reset.value,
#             logging.WARNING: _Color.yellow.value + fmt + _Color.reset.value,
#             logging.ERROR: _Color.red.value + fmt + _Color.reset.value,
#             logging.CRITICAL: _Color.bold_red.value + fmt + _Color.reset.value
#         }
#
#     def format(
#         self,
#         record: logging.LogRecord,
#     ) -> str:
#         log_fmt = self._formats.get(record.levelno)
#         formatter = logging.Formatter(log_fmt)
#
#         return formatter.format(record)


def setup_logger(
    name: str,
    level: int = logging.DEBUG,
    fmt: str = '%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s',
    **kwargs,
) -> logging.Logger:
    formatter = logging.Formatter(
        fmt,
        **extract_kwargs(
            logging.Formatter.__init__,
            **kwargs,
        ),
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    return logger
