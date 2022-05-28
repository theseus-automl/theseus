from collections import namedtuple
from types import MappingProxyType

import numpy as np
from psutil import cpu_count
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
)

_SizeRange = namedtuple(
    '_SizeRange',
    'min max',
)

# everything else is considered as large
_SMALL_DATASET_SIZE_RANGE = _SizeRange(
    min=50,
    max=9999,
)
_MEDIUM_DATASET_SIZE_RANGE = _SizeRange(
    min=10000,
    max=99999,
)

_AFFINITY_PROPAGATION = (
    AffinityPropagation,
    MappingProxyType({
        'max_iter': (1000,),
        'damping': (
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ),
    }),
)


def _get_kmeans_num_clusters(
    dataset_size: int,
) -> tuple:
    nums = []
    exponent = 1
    result = 0

    while result <= dataset_size:
        result = 2 ** exponent
        exponent += 1

        nums.append(result)

    return tuple(nums)


def _prepare_mean_shift(
    dataset_size: int,
) -> tuple:
    min_bin_freq = sorted(set(np.ceil(dataset_size * perc) for perc in np.linspace(0.01, 0.1, num=10)))

    return (
        MeanShift,
        MappingProxyType({
            'max_iter': (1000,),
            'min_bin_freq': tuple(min_bin_freq),
            'n_jobs': (-1,),
        }),
    )


def _prepare_agglomerative(
    is_small: bool,
) -> tuple:
    return (
        AgglomerativeClustering,
        MappingProxyType({
            'affinity': (
                'euclidean',
                'cosine',
            ),
            'distance_threshold': (
                0.5,
                0.7,
                0.9,
            ),
            'linkage': ('ward', 'complete', 'average') if is_small else ('single',),
        }),
    )


def make_param_grid(
    dataset_size: int,
) -> tuple:
    if dataset_size < _SMALL_DATASET_SIZE_RANGE.min:
        raise ValueError('your dataset is too small for clustering algorithms')

    if _SMALL_DATASET_SIZE_RANGE.min <= dataset_size <= _SMALL_DATASET_SIZE_RANGE.max:
        return (
            _prepare_mean_shift(dataset_size),
            _prepare_agglomerative(True),
            _AFFINITY_PROPAGATION,
            (
                KMeans,
                MappingProxyType({
                    'n_clusters': _get_kmeans_num_clusters(dataset_size),
                    'max_iter': (1000,),
                    'algorithm': (
                        'lloyd',
                        'elkan',
                    ),
                }),
            ),
        )

    if _MEDIUM_DATASET_SIZE_RANGE.min <= dataset_size <= _MEDIUM_DATASET_SIZE_RANGE.max:
        return (
            _prepare_agglomerative(False),
            _AFFINITY_PROPAGATION,
            (
                KMeans,
                MappingProxyType({
                    'n_clusters': _get_kmeans_num_clusters(dataset_size),
                    'max_iter': (1000,),
                    'algorithm': ('lloyd',),
                }),
            ),
        )

    return (
        (
            MiniBatchKMeans,
            MappingProxyType({
                'n_clusters': _get_kmeans_num_clusters(dataset_size),
                'max_iter': (1000,),
                'batch_size': (256 * cpu_count(logical=False),),
            }),
        ),
    )
