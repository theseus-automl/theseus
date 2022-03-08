import pandas as pd

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from theseus.dataset.balancing.sampler_mixin import SamplerMixin


class SimilarityUnderSampler(SamplerMixin):
    def __init__(
        self,
        model_name_or_path: str,
    ) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModel.from_pretrained(model_name_or_path)

    def __call__(
        self,
        texts: pd.Series,
        labels: pd.Series,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        df, counts, target_samples = super().__call__(
            texts,
            labels,
            'under',
        )

        return df

    def encode(
        self,
        texts,
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
