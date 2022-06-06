from types import MappingProxyType

from scipy.stats import uniform

TFIDF_GRID = MappingProxyType({
    'ngram_range': (
        (1, 1),
        (1, 2),
        (1, 3),
    ),
    'max_df': uniform(
        loc=0.6,
        scale=0.4
    ),
    'min_df': uniform(
        loc=0,
        scale=0.2,
    ),
    'max_features': (
        100,
        200,
        300,
    ),
})
