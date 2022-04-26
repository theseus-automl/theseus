from types import MappingProxyType

_STRATEGIES = MappingProxyType(
    {
        'under': min,
        'over': max,
    },
)
