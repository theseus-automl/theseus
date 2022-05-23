from types import MappingProxyType

import numpy as np
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    MeanShift,
    OPTICS,
)

CLUSTERERS = (
    (
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
    ),
    (
        MeanShift,
        MappingProxyType({
            'max_iter': (1000,),
            'min_bin_freq': (
                1,
                5,
                10,
            ),
            'n_jobs': (-1,),
        }),
    ),
    (
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
            )
        }),
    ),
    (
        DBSCAN,
        MappingProxyType({
            'eps': (
                0.1,
                0.3,
                0.5,
            ),
            'min_samples': (
                5,
                25,
                50,
                75,
                100,
            ),
            'leaf_size': (
                5,
                25,
                50,
                100,
            ),
            'n_jobs': (-1,),
        }),
    ),
    (
        OPTICS,
        MappingProxyType({
            'min_samples': np.linspace(
                start=0.1,
                stop=1,
                num=10,
            ),
            'max_eps': np.linspace(
                start=0.1,
                stop=0.5,
                num=5,
            ),
            'metric': (
                'euclidean',
                'cosine',
            ),
            'leaf_size': (
                5,
                25,
                50,
                100,
            ),
            'n_jobs': (-1,),
        }),
    ),
    (
        Birch,
        MappingProxyType({
            'threshold': np.linspace(
                start=0.1,
                stop=1,
                num=10,
                endpoint=False,
            ),
        }),
    ),
)
