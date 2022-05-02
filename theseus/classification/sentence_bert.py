from typing import List

import numpy as np

from theseus.classification._abc import EmbeddingsClassifier
from theseus.embedders.bert import BertEmbedder


class SentenceBertClassifier(EmbeddingsClassifier):
    def _embed(
        self,
        texts: List[str],
    ) -> np.ndarray:
        embedder = BertEmbedder(
            self._target_lang,
            self._device,
        )

        return embedder(texts).numpy()
