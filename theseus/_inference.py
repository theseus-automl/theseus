import gc
from typing import (
    Any,
    Callable,
    Optional,
)

import torch

from theseus.exceptions import NotEnoughResourcesError


def gc_with_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def auto_scale_batch_size(
    model: Any,
    example_input: Any,
    initial_batch_size: int,
    eval_func: Optional[Callable] = None,
    *args,
    **kwargs,
) -> int:
    if not _is_power_of_two(initial_batch_size):
        raise ValueError('currently, only powers of two are supported as initial batch size')

    if len(example_input) < initial_batch_size:
        raise ValueError('size of example_input can not be less than initial batch size')

    gc_with_cuda()
    batch_size = initial_batch_size

    while True:
        try:
            if hasattr(model, 'transform'):
                model.transform(example_input[:batch_size])
            elif eval_func is None:
                model(example_input[:batch_size])
            else:
                eval_func(
                    model,
                    example_input[:batch_size],
                    *args,
                    **kwargs,
                )
        except RuntimeError as exception:
            if batch_size > 1 and _should_reduce_batch_size(exception):
                batch_size //= 2

                gc_with_cuda()
            else:
                raise NotEnoughResourcesError('unable to find batch size due to memory issues')
        else:
            return batch_size


def _is_power_of_two(
    num: int,
) -> bool:
    return num > 0 and num & (num - 1) == 0


def _should_reduce_batch_size(
    exception: Exception,
) -> bool:
    return _is_cuda_out_of_memory(exception) or _is_cudnn_snafu(exception) or _is_out_of_cpu_memory(exception)


def _is_cuda_out_of_memory(
    exception: Exception,
) -> bool:
    return (
        isinstance(exception, RuntimeError) and len(exception.args) == 1 and 'CUDA out of memory' in exception.args[0]
    )


def _is_cudnn_snafu(
    exception: Exception,
) -> bool:
    # For / because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and 'cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.' in exception.args[0]
    )


def _is_out_of_cpu_memory(
    exception,
) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and 'DefaultCPUAllocator: can\'t allocate memory' in exception.args[0]
    )
