import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class DenseTfidfVectorizer(TfidfVectorizer):
    def __init__(
        self,
        **kwargs,
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
            **kwargs,
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
