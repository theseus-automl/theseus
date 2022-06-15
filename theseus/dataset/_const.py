from types import MappingProxyType

_STRATEGIES = MappingProxyType(
    {
        'under': min,
        'over': max,
    },
)
_BALANCING_THRESHOLD = MappingProxyType(
    {
        'under': 1.5,
        'over': 0.5,
    },
)
