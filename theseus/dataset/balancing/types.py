from typing import Union

from theseus.dataset.balancing.augmentation import AugmentationOverSampler
from theseus.dataset.balancing.random import (
    RandomOverSampler,
    RandomUnderSampler,
)
from theseus.dataset.balancing.similarity import (
    SimilarityOverSampler,
    SimilarityUnderSampler,
)

SamplerType = Union[
    RandomUnderSampler,
    RandomOverSampler,
    SimilarityUnderSampler,
    SimilarityOverSampler,
    AugmentationOverSampler,
]
