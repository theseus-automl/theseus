from abc import (
    ABC,
    abstractmethod,
)
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


class _SimilaritySampler(_Sampler, ABC):
    def __init__(
        self,
        model_name_or_path: str,
        strategy: str,
    ) -> NoReturn:
        super().__init__(strategy)
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModel.from_pretrained(model_name_or_path)

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
        model_name_or_path: str,
    ) -> NoReturn:
        super().__init__(
            model_name_or_path,
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
        model_name_or_path: str,
    ) -> NoReturn:
        super().__init__(
            model_name_or_path,
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
