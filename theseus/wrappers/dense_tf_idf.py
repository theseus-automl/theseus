import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class DenseTfidfVectorizer(TfidfVectorizer):
    def __init__(
        self,
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
    ) -> None:
        super().__init__(
            input='content',
            encoding='utf-8',
            decode_error='ignore',
            binary=False,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            dtype=np.float64,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
        )

    def transform(
        self,
        raw_documents,
    ) -> np.ndarray:
        return super().transform(raw_documents).toarray()

    def fit_transform(
        self,
        raw_documents,
        y=None,
    ) -> np.ndarray:
        return super().fit_transform(raw_documents, y=y).toarray()
