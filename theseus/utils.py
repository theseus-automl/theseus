from typing import (
    Generator,
    Collection,
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
