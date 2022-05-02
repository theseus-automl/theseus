from typing import List

import numpy as np

from theseus.classification._abc import EmbeddingsClassifier
from theseus.embedders.fast_text import FasttextEmbedder


class FastTextClassifier(EmbeddingsClassifier):
    def _embed(
        self,
        texts: List[str],
    ) -> np.ndarray:
        embedder = FasttextEmbedder(self._target_lang)

        return embedder(texts)
