from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Union,
)


def add_param_grid_prefix(
    param_grid: Union[dict, MappingProxyType],
    prefix: str,
) -> Dict[str, Any]:
    prefixed = {}

    for name, options in param_grid.items():
        prefixed[f'{prefix}__{name}'] = options

    return prefixed
