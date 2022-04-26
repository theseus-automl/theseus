from inspect import signature
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generator,
)

import numpy as np


def chunkify(
    it: Collection,
    num_chunks: int,
) -> Generator[Collection, None, None]:
    if num_chunks <= 0:
        raise ValueError('num_chunks must be greater than zero')

    if not len(it):
        raise ValueError('unable to chunkify empty iterable')
    elif len(it) < num_chunks:
        yield [it]
    else:
        for chunk in np.array_split(it, num_chunks):
            chunk = chunk.tolist()

            yield chunk


def extract_kwargs(
    func: Callable,
    **kwargs: Any,
) -> Dict[str, Any]:
    args_names = [k for k, v in signature(func).parameters.items()]

    return {k: kwargs.pop(k) for k in dict(kwargs) if k in args_names}
