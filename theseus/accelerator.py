import inspect
import subprocess
import warnings
from io import StringIO
from typing import (
    List,
    Optional,
    Union,
)

import pandas as pd
import torch

from theseus.exceptions import DeviceError
from theseus.log import setup_logger
from theseus.validators.one_of import OneOf

_ACCELERATORS = frozenset((
    'cpu',
    'gpu',
    'tpu',
    'ipu',
    'hpu',
    'auto',
))
_BACKENDS = frozenset((
    'native',
    'apex',
))
_AMP_LEVELS = frozenset((
    'O0',
    'O1',
    'O2',
    'O3',
    None,
))
_PRECISIONS = frozenset((
    16,
    32,
    64,
))

_logger = setup_logger(__name__)
DeviceList = List[Union[int, str]]


class Accelerator:
    accelerator = OneOf(_ACCELERATORS)
    amp_backend = OneOf(_BACKENDS)
    amp_level = OneOf(_AMP_LEVELS)

    def __init__(
        self,
        accelerator: str = 'auto',
        amp_backend: str = 'native',
        amp_level: Optional[str] = None,
        auto_select_gpus: bool = True,
        gpus: Optional[Union[DeviceList, int]] = None,
        tpu_cores: Optional[int] = None,
        ipus: Optional[int] = None,
    ) -> None:
        if amp_backend == 'apex' and amp_level is None:
            raise ValueError('apex backend was chosen, but no level was provided')

        if accelerator == 'tpu' and tpu_cores is None:
            raise ValueError('tpu_cores param not provided when using tpu accelerator')

        if accelerator == 'ipu' and ipus is None:
            raise ValueError('ipus param not provided when using ipu accelerator')

        if gpus is not None and auto_select_gpus:
            warnings.warn(
                'both gpus and auto_select_gpus were provided, using gpus param',
                UserWarning,
            )
            self.auto_select_gpus = False
        else:
            self.auto_select_gpus = True

        self.accelerator = accelerator
        self.amp_backend = amp_backend
        self.amp_level = amp_level
        self.gpus = gpus
        self.tpu_cores = tpu_cores
        self.ipus = ipus

        try:
            self._validate_gpus()
        except DeviceError as err:
            _logger.error(f'device error: {err}')
            raise

    def to_dict(
        self,
    ) -> dict:
        params = {}

        for name in dir(self):
            value = getattr(
                self,
                name,
            )

            if not name.startswith('__') and not inspect.ismethod(value):
                params[name] = value

        return params

    def select_single_gpu(
        self,
    ) -> torch.device:
        device_error = DeviceError('no suitable GPU was found')

        if self.gpus is None:
            try:
                gpu_stats = subprocess.check_output([
                    'nvidia-smi',
                    '--format=csv',
                    '--query-gpu=memory.used,memory.free',
                ])
            except FileNotFoundError:
                raise DeviceError('nvidia-smi not available')

            gpu_df = pd.read_csv(
                StringIO(gpu_stats),
                names=[
                    'memory.used',
                    'memory.free',
                ],
                skiprows=1,
            )
            _logger.debug(f'GPU usage:\n{gpu_df}')

            gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
            idx = gpu_df['memory.free'].idxmax()
            free_mem = gpu_df.iloc[idx]['memory.free']
            _logger.info(f'Picked GPU{idx} with {free_mem} free MiB')

            return torch.device(idx)

        for gpu in self.gpus:
            device = torch.device(f'cuda:{gpu}')

            try:
                torch.tensor(
                    [1],
                    device=device,
                )
            except (AssertionError, RuntimeError):
                raise device_error

            return device

        raise device_error

    def _validate_gpus(
        self,
    ) -> None:
        if isinstance(self.gpus, list):
            for gpu in self.gpus:
                try:
                    torch.cuda.get_device_name(gpu)
                except (AssertionError, RuntimeError):
                    raise DeviceError(f'invalid device {gpu}')
