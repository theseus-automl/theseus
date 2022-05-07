from types import MappingProxyType
from typing import (
    Any,
    Generator,
    Iterable,
    Sequence,
    Union,
)

import torch
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from torch.nn import functional as F  # noqa: WPS111, WPS347, N812
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput

from theseus._inference import auto_scale_batch_size
from theseus.exceptions import (
    NotEnoughResourcesError,
    UnsupportedLanguageError,
)
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger

_MULTILANG_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
SBERT_SUPPORTED_LANGS = MappingProxyType({
    # Mono-language models
    LanguageCode.ENGLISH: 'sentence-transformers/all-mpnet-base-v2',

    # Multilanguage models
    LanguageCode.ARABIC: _MULTILANG_MODEL,
    LanguageCode.BULGARIAN: _MULTILANG_MODEL,
    LanguageCode.CATALAN: _MULTILANG_MODEL,
    LanguageCode.CZECH: _MULTILANG_MODEL,
    LanguageCode.DANISH: _MULTILANG_MODEL,
    LanguageCode.GERMAN: _MULTILANG_MODEL,
    LanguageCode.GREEK: _MULTILANG_MODEL,
    LanguageCode.SPANISH: _MULTILANG_MODEL,
    LanguageCode.ESTONIAN: _MULTILANG_MODEL,
    LanguageCode.PERSIAN: _MULTILANG_MODEL,
    LanguageCode.FINNISH: _MULTILANG_MODEL,
    LanguageCode.FRENCH: _MULTILANG_MODEL,
    LanguageCode.GALICIAN: _MULTILANG_MODEL,
    LanguageCode.GUJARATI: _MULTILANG_MODEL,
    LanguageCode.HEBREW: _MULTILANG_MODEL,
    LanguageCode.HINDI: _MULTILANG_MODEL,
    LanguageCode.CROATIAN: _MULTILANG_MODEL,
    LanguageCode.HUNGARIAN: _MULTILANG_MODEL,
    LanguageCode.ARMENIAN: _MULTILANG_MODEL,
    LanguageCode.INDONESIAN: _MULTILANG_MODEL,
    LanguageCode.ITALIAN: _MULTILANG_MODEL,
    LanguageCode.JAPANESE: _MULTILANG_MODEL,
    LanguageCode.GEORGIAN: _MULTILANG_MODEL,
    LanguageCode.KOREAN: _MULTILANG_MODEL,
    LanguageCode.KURDISH: _MULTILANG_MODEL,
    LanguageCode.LITHUANIAN: _MULTILANG_MODEL,
    LanguageCode.LATVIAN: _MULTILANG_MODEL,
    LanguageCode.MACEDONIAN: _MULTILANG_MODEL,
    LanguageCode.MONGOLIAN: _MULTILANG_MODEL,
    LanguageCode.MARATHI: _MULTILANG_MODEL,
    LanguageCode.MALAY: _MULTILANG_MODEL,
    LanguageCode.BURMESE: _MULTILANG_MODEL,
    LanguageCode.NORWEGIAN: _MULTILANG_MODEL,
    LanguageCode.DUTCH: _MULTILANG_MODEL,
    LanguageCode.POLISH: _MULTILANG_MODEL,
    LanguageCode.PORTUGUESE: _MULTILANG_MODEL,
    LanguageCode.ROMANIAN: _MULTILANG_MODEL,
    LanguageCode.RUSSIAN: _MULTILANG_MODEL,
    LanguageCode.SLOVAK: _MULTILANG_MODEL,
    LanguageCode.SLOVENIAN: _MULTILANG_MODEL,
    LanguageCode.ALBANIAN: _MULTILANG_MODEL,
    LanguageCode.SERBIAN: _MULTILANG_MODEL,
    LanguageCode.SWEDISH: _MULTILANG_MODEL,
    LanguageCode.THAI: _MULTILANG_MODEL,
    LanguageCode.TURKISH: _MULTILANG_MODEL,
    LanguageCode.UKRAINIAN: _MULTILANG_MODEL,
    LanguageCode.URDU: _MULTILANG_MODEL,
    LanguageCode.VIETNAMESE: _MULTILANG_MODEL,
    LanguageCode.CHINESE: _MULTILANG_MODEL,
})

_CLAMP_MIN = 1e-9
_BATCH_SIZE_SEARCH_START = 32768

_logger = setup_logger(__name__)


class BertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_lang: LanguageCode,
        device: torch.device,
    ) -> None:
        if target_lang not in SBERT_SUPPORTED_LANGS:
            raise UnsupportedLanguageError(f'BERT embeddings are not available for {target_lang}')

        self.target_lang = target_lang
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(SBERT_SUPPORTED_LANGS[self.target_lang])
        self._model = AutoModel.from_pretrained(SBERT_SUPPORTED_LANGS[self.target_lang]).to(self.device)

        if self.device.type == 'cuda':
            try:
                self._effective_batch_size = auto_scale_batch_size(
                    self,
                    ['a '.strip() * self._tokenizer.model_max_length for _ in range(_BATCH_SIZE_SEARCH_START)],
                    _BATCH_SIZE_SEARCH_START,
                )
            except NotEnoughResourcesError as err:
                _logger.error(err)
                raise
        else:
            self._effective_batch_size = 1

    def fit(
        self,
        texts: Union[str, Iterable[str]],
        y: Any = None,
    ) -> 'BertEmbedder':
        return self

    def transform(
        self,
        texts: Union[str, Iterable[str]],
        y: Any = None,
    ) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        if len(texts) <= self._effective_batch_size:
            return self._encode(texts)

        embeddings = []

        for batch in self._make_batches(texts, self._effective_batch_size):
            embeddings.append(self._encode(batch))

        return torch.stack(embeddings)

    def _encode(
        self,
        texts: Iterable[str],
    ) -> torch.Tensor:
        encoded_input = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )

        with torch.no_grad():
            model_output = self._model(
                **encoded_input,
                return_dict=True,
            )

        embeddings = self._mean_pooling(
            model_output,
            encoded_input['attention_mask'],
        )
        return F.normalize(
            embeddings,
            p=2,
            dim=1,
        )

    @staticmethod
    def _make_batches(
        inp: Sequence,
        batch_size: int,
    ) -> Generator[Sequence, None, None]:
        yield from (
            inp[i:i + batch_size]
            for i in range(0, len(inp), batch_size)
        )

    @staticmethod
    def _mean_pooling(
        model_output: BaseModelOutput,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        clamped = torch.clamp(
            input_mask_expanded.sum(1),
            min=_CLAMP_MIN,
        )

        return torch.sum(token_embeddings * input_mask_expanded, 1) / clamped
