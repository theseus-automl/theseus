from pathlib import Path

from theseus.classification._param_grids import TFIDF_GRID
from theseus.classification.embeddings import EmbeddingsClassifier
from theseus.wrappers.dense_tf_idf import DenseTfidfVectorizer
from theseus.lang_code import LanguageCode


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
