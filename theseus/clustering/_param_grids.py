from types import MappingProxyType

import numpy as np
from psutil import cpu_count
from scipy.stats import (
    randint,
    uniform,
)
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
        'damping': uniform(
            loc=0.5,
            scale=0.9 - 0.5,
        ),
    }),
)


def _get_kmeans_num_clusters(
    dataset_size: int,
):
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
            'min_bin_freq': randint(
                low=min_bin_freq[0],
                high=min_bin_freq[-1],
            ),
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
            'distance_threshold': uniform(
                loc=0.5,
                scale=0.9 - 0.5,
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
            (
                MiniBatchKMeans,
                MappingProxyType({
                    'n_clusters': _get_kmeans_num_clusters(dataset_size),
                    'max_iter': (500,),
                    'batch_size': (256 * cpu_count(logical=False),),
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
