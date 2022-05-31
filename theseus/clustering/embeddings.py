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
    calinski_harabasz_score,
    davies_bouldin_score,
    make_scorer,
    silhouette_score,
)

from theseus.abc.embeddings_estimator import EmbeddingsEstimator
from theseus.clustering._param_grids import make_param_grid
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

CLUSTERIZATION_METRICS = MappingProxyType({
    'silhouette': make_scorer(silhouette_score),
    'chs': make_scorer(calinski_harabasz_score),
    'dbs': make_scorer(davies_bouldin_score),
})


class EmbeddingsClusterer(EmbeddingsEstimator, ABC):
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
            make_param_grid,
            dict(CLUSTERIZATION_METRICS),
            'silhouette',
            embedder_param_grid,
            supported_languages,
        )


class FastTextClusterer(EmbeddingsClusterer):
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


class SentenceBertClusterer(EmbeddingsClusterer):
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


class TfIdfClusterer(EmbeddingsClusterer):
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
