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

from theseus.cv import (
    MEDIUM_DATASET_SIZE_RANGE,
    SMALL_DATASET_SIZE_RANGE,
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
        nums.append(result)
        result = 2 ** exponent
        exponent += 1

    return tuple(nums[1:])


def _prepare_mean_shift(
    dataset_size: int,
) -> tuple:
    min_bin_freq = sorted(set(np.ceil(dataset_size * perc) for perc in np.linspace(0.01, 0.1, num=10)))

    return (
        MeanShift,
        MappingProxyType({
            'max_iter': (500,),
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
            'n_clusters': (None,),
        }),
    )


def make_param_grid(
    dataset_size: int,
) -> tuple:
    if dataset_size < SMALL_DATASET_SIZE_RANGE.min:
        raise ValueError('your dataset is too small for clustering algorithms')

    if SMALL_DATASET_SIZE_RANGE.min <= dataset_size <= SMALL_DATASET_SIZE_RANGE.max:
        return (
            _prepare_mean_shift(dataset_size),
            _prepare_agglomerative(True),
            _AFFINITY_PROPAGATION,
            (
                KMeans,
                MappingProxyType({
                    'n_clusters': _get_kmeans_num_clusters(dataset_size),
                    'max_iter': (300,),
                    'algorithm': (
                        'lloyd',
                        'elkan',
                    ),
                }),
            ),
        )

    if MEDIUM_DATASET_SIZE_RANGE.min <= dataset_size <= MEDIUM_DATASET_SIZE_RANGE.max:
        return (
            _prepare_agglomerative(False),
            _AFFINITY_PROPAGATION,
            (
                KMeans,
                MappingProxyType({
                    'n_clusters': _get_kmeans_num_clusters(dataset_size),
                    'max_iter': (500,),
                    'algorithm': ('lloyd',),
                }),
            ),
        )

    return (
        (
            MiniBatchKMeans,
            MappingProxyType({
                'n_clusters': _get_kmeans_num_clusters(dataset_size),
                'max_iter': (500,),
                'batch_size': (256 * cpu_count(logical=False),),
            }),
        ),
    )
