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

from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.dataset.balancing._sampler import (
    _prepare,
    _Sampler,
)


SIMILARITY_MODELS = MappingProxyType(
    {
        # Mono-language models
        LanguageCode.ENGLISH: 'sentence-transformers/all-mpnet-base-v2',

        # Multilanguage models
        LanguageCode.ARABIC: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.BULGARIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.CATALAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.CZECH: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.DANISH: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.GERMAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.GREEK: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.SPANISH: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.ESTONIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.PERSIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.FINNISH: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.FRENCH: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.GALICIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.GUJARATI: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.HEBREW: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.HINDI: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.CROATIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.HUNGARIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.ARMENIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.INDONESIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.ITALIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.JAPANESE: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.GEORGIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.KOREAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.KURDISH: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.LITHUANIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.LATVIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.MACEDONIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.MONGOLIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.MARATHI: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.MALAY: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.BURMESE: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.NORWEGIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.DUTCH: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.POLISH: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.PORTUGUESE: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.ROMANIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.RUSSIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.SLOVAK: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.SLOVENIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.ALBANIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.SERBIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.SWEDISH: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.THAI: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.TURKISH: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.UKRAINIAN: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.URDU: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.VIETNAMESE: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        LanguageCode.CHINESE: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
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
        texts: pd.Series,
        labels: pd.Series,
    ) -> pd.DataFrame:
        df, counts, target_samples = _prepare(
            texts,
            labels,
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

        return df

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
