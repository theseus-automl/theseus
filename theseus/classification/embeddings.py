from abc import ABC
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Optional,
)

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)

from theseus.abc.embeddings_estimator import EmbeddingsEstimator
from theseus.classification._param_grids import CLASSIFIERS
from theseus.embedders.bert import (
    BertEmbedder,
    SBERT_SUPPORTED_LANGS,
)
from theseus.embedders.fast_text import (
    FasttextEmbedder,
    FT_SUPPORTED_LANGS,
)
from theseus.lang_code import LanguageCode
from theseus.param_grids import TFIDF_GRID
from theseus.wrappers.dense_tf_idf import DenseTfidfVectorizer

CLASSIFICATION_METRICS = MappingProxyType({
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
})


class EmbeddingsClassifier(EmbeddingsEstimator, ABC):
    def __init__(
        self,
        target_lang: LanguageCode,
        out_dir: Path,
        embedder: Any,
        embedder_param_grid: Optional[Dict[str, Any]] = None,
        supported_languages: Optional[MappingProxyType] = None,
    ) -> None:
        super().__init__(
            target_lang,
            out_dir,
            embedder,
            CLASSIFIERS,
            dict(CLASSIFICATION_METRICS),
            'f1',
            embedder_param_grid,
            supported_languages,
        )


class FastTextClassifier(EmbeddingsClassifier):
    def __init__(
        self,
        target_lang: LanguageCode,
        out_dir: Path,
    ) -> None:
        super().__init__(
            target_lang,
            out_dir,
            FasttextEmbedder(target_lang),
            supported_languages=FT_SUPPORTED_LANGS,
        )


class SentenceBertClassifier(EmbeddingsClassifier):
    def __init__(
        self,
        target_lang: LanguageCode,
        out_dir: Path,
        device: torch.device,
    ) -> None:
        super().__init__(
            target_lang,
            out_dir,
            BertEmbedder(
                target_lang,
                device,
            ),
            supported_languages=SBERT_SUPPORTED_LANGS,
        )


class TfIdfClassifier(EmbeddingsClassifier):
    def __init__(
        self,
        target_lang: LanguageCode,
        out_dir: Path,
    ) -> None:
        super().__init__(
            target_lang,
            out_dir,
            DenseTfidfVectorizer(),
            TFIDF_GRID,
        )
