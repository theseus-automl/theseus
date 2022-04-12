from abc import (
    ABC,
    abstractmethod,
)
from types import MappingProxyType
from typing import (
    List,
    NoReturn,
)

import pandas as pd
import torch
from torch.nn import functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
)

from theseus.dataset.balancing._sampler import (
    _prepare,
    _Sampler,
)
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode

_MULTILANG_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
SIMILARITY_MODELS = MappingProxyType(
    {
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
    },
)


class _SimilaritySampler(_Sampler, ABC):
    def __init__(
        self,
        target_lang: LanguageCode,
        strategy: str,
    ) -> None:
        super().__init__(strategy)

        if target_lang not in SIMILARITY_MODELS:
            raise UnsupportedLanguageError(f'sentence similarity is not available for language {target_lang}')

        self._tokenizer = AutoTokenizer.from_pretrained(SIMILARITY_MODELS[target_lang])
        self._model = AutoModel.from_pretrained(SIMILARITY_MODELS[target_lang])

    def __call__(
        self,
        dataset: TextDataset,
    ) -> TextDataset:
        df, counts, target_samples = _prepare(
            dataset.texts,
            dataset.labels,
            self._strategy,
        )

        for label, n_samples in counts.items():
            if n_samples != target_samples:
                index = df[df['labels'] == label].index
                encoded = self._encode(df.loc[index]['texts'].tolist())
                cosine = self._cosine_similarity(encoded)

                df = self._update(
                    df,
                    index,
                    cosine,
                    abs(n_samples - target_samples),
                )

        return TextDataset(
            df['texts'].tolist(),
            df['labels'].tolist(),
        )

    @staticmethod
    @abstractmethod
    def _update(
        df: pd.DataFrame,
        index: pd.Index,
        cosine: torch.Tensor,
        k: int,
    ) -> NoReturn:
        raise NotImplementedError

    def _encode(
        self,
        texts: List[str],
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
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        return embeddings

    @staticmethod
    def _mean_pooling(
        model_output,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def _cosine_similarity(
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        similarity = torch.matmul(
            encoded,
            encoded.T,
        )
        square_mag = torch.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[torch.isinf(inv_square_mag)] = 0
        inv_mag = torch.sqrt(inv_square_mag)

        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag

        return cosine


class SimilarityUnderSampler(_SimilaritySampler):
    def __init__(
        self,
        target_lang: LanguageCode,
    ) -> NoReturn:
        super().__init__(
            target_lang,
            'under',
        )

    @staticmethod
    def _update(
        df: pd.DataFrame,
        index: pd.Index,
        cosine: torch.Tensor,
        k: int,
    ) -> pd.DataFrame:
        local_index = torch.topk(
            torch.sum(
                cosine,
                dim=0,
            ),
            k=k,
        ).indices.numpy()
        df = df.drop(index[local_index])

        return df


class SimilarityOverSampler(_SimilaritySampler):
    def __init__(
        self,
        target_lang: LanguageCode,
    ) -> NoReturn:
        super().__init__(
            target_lang,
            'over',
        )

    @staticmethod
    def _update(
        df: pd.DataFrame,
        index: pd.Index,
        cosine: torch.Tensor,
        k: int,
    ) -> pd.DataFrame:
        local_index = torch.topk(
            torch.sum(
                cosine,
                dim=0,
            ),
            k=k,
            largest=False,
        ).indices.numpy()

        df = pd.concat(
            [
                df,
                df.loc[index[local_index]],
            ],
            ignore_index=True,
        )

        return df
