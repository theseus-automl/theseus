from pathlib import Path

from theseus.classification.embeddings import EmbeddingsClassifier
from theseus.embedders.fast_text import (
    _SUPPORTED_LANGS,
    FasttextEmbedder,
)
from theseus.lang_code import LanguageCode


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
            supported_languages=_SUPPORTED_LANGS,
        )
