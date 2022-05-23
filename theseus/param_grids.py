from types import MappingProxyType

import numpy as np

TFIDF_GRID = MappingProxyType({
    'ngram_range': (
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 3),
    ),
    'max_df': np.linspace(
        start=0.6,
        stop=1,
        num=10,
    ),
    'min_df': np.linspace(
        start=0,
        stop=0.2,
        num=5,
    ),
    'max_features': (
        100,
        200,
        300,
    ),
})