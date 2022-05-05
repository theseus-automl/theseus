import os
from pathlib import Path
from typing import (
    Dict,
    Union,
)

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.nn import functional as F  # noqa: WPS111, WPS347, N812
from torch.optim import (
    AdamW,
    Optimizer,
)
from torch.utils.data import DataLoader
from torchmetrics import (
    Accuracy,
    F1Score,
    Metric,
    Precision,
    Recall,
)
from transformers import (
    BatchEncoding,
    BertForSequenceClassification,
    BertTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from theseus.dataset.text_dataset import TextDataset


class BertForClassification(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        num_labels: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._model = BertForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
        )
        self._tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

        self.lr = None
        self.batch_size = None
        self._train_dataset = None
        self._val_dataset = None
        self._class_weights = None

        self.metrics = {
            'train': self._make_metrics(),
            'val': self._make_metrics(),
        }

    def set_data(
        self,
        dataset: TextDataset,
    ) -> None:
        train_indices, val_indices = train_test_split(
            list(range(len(dataset.labels))),
            test_size=0.2,
            stratify=dataset.labels,
        )
        self._train_dataset = torch.utils.data.Subset(
            dataset,
            train_indices,
        )
        self._val_dataset = torch.utils.data.Subset(
            dataset,
            val_indices,
        )
        self._class_weights = dataset.class_weights

    def configure_optimizers(
        self,
    ) -> Optimizer:
        return AdamW(
            self._model.parameters(),
            lr=self.lr or self.learning_rate,
            amsgrad=True,
        )

    def forward(
        self,
        inputs: BatchEncoding,
    ) -> SequenceClassifierOutput:
        return self._model(**inputs)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        labels = batch.pop('Class')
        logits = self._model(**batch).logits
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self._class_weights,
        )
        predictions = logits.argmax(axis=1)

        self.log(
            'train/loss',
            loss.item(),
        )
        self._calculate_metrics(
            predictions,
            labels,
            'train',
        )

        return loss

    def training_epoch_end(
        self,
        outputs: SequenceClassifierOutput,
    ) -> None:
        self._reset_metrics('train')

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        labels = batch.pop('Class')
        logits = self._model(**batch).logits
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(axis=1)

        self.log('val/loss', loss.item())
        self._calculate_metrics(
            predictions,
            labels,
            'val',
        )

    def validation_epoch_end(
        self,
        outputs: SequenceClassifierOutput,
    ) -> None:
        self._reset_metrics('val')

    def train_dataloader(
        self,
    ) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            shuffle=True,
            num_workers=os.cpu_count(),
            collate_fn=self._collate_fn,
        )

    def val_dataloader(
        self,
    ) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            shuffle=False,
            num_workers=os.cpu_count(),
            collate_fn=self._collate_fn,
        )

    def _calculate_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        prefix: str,
    ) -> None:
        for name, metric in self.metrics[prefix]:
            self.log(
                f'{prefix}/name',
                metric(
                    predictions,
                    labels,
                ),
            )

    def _reset_metrics(
        self,
        prefix: str,
    ) -> None:
        for metric in self.metrics[prefix]:
            metric.reset()

    @staticmethod
    def _make_metrics() -> Dict[str, Metric]:
        return {
            'accuracy': Accuracy(),
            'f1': F1Score(),
            'precision': Precision(),
            'recall': Recall(),
        }

    def _collate_fn(
        self,
        input_data,
    ) -> BatchEncoding:
        texts, labels = zip(*input_data)
        labels = torch.LongTensor(labels)

        inputs = self._tokenizer(
            texts,
            return_tensors='pt',
            padding='longest',
            max_length=256,
            truncation=True,
        )
        inputs['Class'] = labels

        return inputs
