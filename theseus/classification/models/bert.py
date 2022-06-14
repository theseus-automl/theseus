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

from theseus.cv import select_test_size
from theseus.dataset.text_dataset import TextDataset

_START_LR = 1e-8


class BertForClassification(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        num_labels: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._num_labels = num_labels

        self._model = BertForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=self._num_labels,
        )
        self._model.classifier.weight.data.normal_(
            mean=0.0,
            std=0.02,
        )
        self._model.classifier.weight.requires_grad = True
        self._model.classifier.bias.data.zero_()
        self._model.classifier.bias.requires_grad = True

        self._tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

        self.learning_rate = _START_LR
        self.batch_size = 1
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
            test_size=select_test_size(len(dataset)),
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
            lr=self.learning_rate,
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
        for name, metric in self.metrics['train'].items():
            self.log(
                f'train/{name}',
                metric.compute().item(),
            )

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
        for name, metric in self.metrics['val'].items():
            self.log(
                f'train/{name}',
                metric.compute().item(),
            )

    def train_dataloader(
        self,
    ) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            shuffle=True,
            num_workers=os.cpu_count(),
            collate_fn=self._collate_fn,
            batch_size=self.batch_size,
        )

    def val_dataloader(
        self,
    ) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            shuffle=False,
            num_workers=os.cpu_count(),
            collate_fn=self._collate_fn,
            batch_size=self.batch_size,
        )

    def _calculate_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        prefix: str,
    ) -> None:
        for metric in self.metrics[prefix].values():
            metric.update(
                predictions.clone().detach().cpu(),
                labels.clone().detach().cpu(),
            )
            # self.log(
            #     f'{prefix}/{name}',
            #     metric(
            #         predictions.clone().detach().cpu(),
            #         labels.clone().detach().cpu(),
            #     ),
            # )

    def _reset_metrics(
        self,
        prefix: str,
    ) -> None:
        for metric in self.metrics[prefix].values():
            metric.reset()

    def _make_metrics(
        self,
    ) -> Dict[str, Metric]:
        return {
            'accuracy': Accuracy(average='weighted', num_classes=self._num_labels),
            'f1': F1Score(average='weighted', num_classes=self._num_labels),
            'precision': Precision(average='weighted', num_classes=self._num_labels),
            'recall': Recall(average='weighted', num_classes=self._num_labels),
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
            max_length=self._tokenizer.model_max_length,
            truncation=True,
        )
        inputs['Class'] = labels

        return inputs
