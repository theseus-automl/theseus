import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class DenseTfidfVectorizer(TfidfVectorizer):
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
