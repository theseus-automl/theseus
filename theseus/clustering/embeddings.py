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


def _modified_silhouette_score(
    estimator,
    data,
) -> float:
    estimator.fit_predict(data)

    try:
        return silhouette_score(
            estimator['emb'].transform(data),
            estimator['clf'].labels_,
        )
    except ValueError:
        return -1.0


def _modified_calinski_harabasz_score(
    estimator,
    data,
) -> float:
    estimator.fit_predict(data)

    try:
        return calinski_harabasz_score(
            estimator['emb'].transform(data),
            estimator['clf'].labels_,
        )
    except ValueError:
        return 0.0


def _modified_davies_bouldin_score(
    estimator,
    data,
) -> float:
    estimator.fit_predict(data)

    try:
        return davies_bouldin_score(
            estimator['emb'].transform(data),
            estimator['clf'].labels_,
        )
    except ValueError:
        return 1e+9


CLUSTERIZATION_METRICS = MappingProxyType({
    'silhouette': _modified_silhouette_score,
    'chs': _modified_calinski_harabasz_score,
    'dbs': _modified_davies_bouldin_score,
})


class EmbeddingsClusterer(EmbeddingsEstimator, ABC):
    def __init__(
        self,
        target_lang: LanguageCode,
        out_dir: Path,
        embedder: Any,
        embedder_param_grid: Optional[Dict[str, Any]] = None,
        supported_languages: Optional[MappingProxyType] = None,
        n_jobs: int = -1,
        n_iter: int = 5,
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
            n_jobs,
            n_iter,
        )


class FastTextClusterer(EmbeddingsClusterer):
    def __init__(
        self,
        target_lang: LanguageCode,
        out_dir: Path,
        n_jobs: int = -1,
        n_iter: int = 5,
    ) -> None:
        super().__init__(
            target_lang,
            out_dir,
            FasttextEmbedder(target_lang),
            supported_languages=FT_SUPPORTED_LANGS,
            n_jobs=n_jobs,
            n_iter=n_iter,
        )


class SentenceBertClusterer(EmbeddingsClusterer):
    def __init__(
        self,
        target_lang: LanguageCode,
        out_dir: Path,
        device: torch.device,
        n_jobs: int = -1,
        n_iter: int = 5,
    ) -> None:
        super().__init__(
            target_lang,
            out_dir,
            BertEmbedder(
                target_lang,
                device,
            ),
            supported_languages=SBERT_SUPPORTED_LANGS,
            n_jobs=n_jobs,
            n_iter=n_iter,
        )


class TfIdfClusterer(EmbeddingsClusterer):
    def __init__(
        self,
        target_lang: LanguageCode,
        out_dir: Path,
        n_jobs: int = -1,
        n_iter: int = 5,
    ) -> None:
        super().__init__(
            target_lang,
            out_dir,
            DenseTfidfVectorizer(),
            TFIDF_GRID,
            n_jobs=n_jobs,
            n_iter=n_iter,
        )
