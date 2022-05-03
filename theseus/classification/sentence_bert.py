from pathlib import Path

import torch

from theseus.classification.embeddings import EmbeddingsClassifier
from theseus.embedders.bert import (
    _SUPPORTED_LANGS,
    BertEmbedder,
)
from theseus.lang_code import LanguageCode


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
            supported_languages=_SUPPORTED_LANGS,
        )
