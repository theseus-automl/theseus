from pathlib import Path
from types import MappingProxyType
from typing import (
    Dict,
    Tuple,
)

import pytorch_lightning as pl

from theseus._inference import gc_with_cuda
from theseus.accelerator import Accelerator
from theseus.classification.models.bert import BertForClassification
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.plotting.display import save_fig

_SUPPORTED_LANGS = MappingProxyType({
    LanguageCode.RUSSIAN: 'DeepPavlov/rubert-base-cased',
})

_MAX_EPOCHS = 2


class BertClassifier:
    def __init__(
        self,
        target_lang: LanguageCode,
        num_labels: int,
        out_dir: Path,
        accelerator: Accelerator,
    ) -> None:
        if target_lang not in _SUPPORTED_LANGS:
            raise UnsupportedLanguageError(f'BERT classifier is unavailable for {target_lang}')

        self._model = BertForClassification(
            _SUPPORTED_LANGS[target_lang],
            num_labels,
        )
        self._out_dir = out_dir
        self._accelerator_params = accelerator.to_dict()

    def fit(
        self,
        dataset: TextDataset,
    ) -> Tuple[float, Dict[str, float], float, Dict[str, float]]:
        self._model.set_data(dataset)

        logger = pl.loggers.TensorBoardLogger(
            save_dir=self._out_dir / 'logs',
            name='BERT Classifier',
        )
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=_MAX_EPOCHS,
            gradient_clip_val=2,
            gradient_clip_algorithm='norm',
            deterministic=True,
            precision=16,
            accumulate_grad_batches=2048,
            log_every_n_steps=8,
            **self._accelerator_params,
        )

        lr_finder = trainer.tuner.lr_find(
            self._model,
            min_lr=1e-8,
            max_lr=1e-3,
            update_attr=True,
        )
        save_fig(
           self._out_dir / 'lr_tuner.png',
           False,
           lr_finder.plot(suggest=True),
        )
        trainer.tuner.scale_batch_size(self._model)
        gc_with_cuda()

        trainer.reset_train_dataloader(self._model)
        trainer.reset_val_dataloader(self._model)
        trainer.fit(self._model)

        train_metrics = {name: metric.compute().item() for name, metric in self._model.metrics['train'].items()}
        val_metrics = {name: metric.compute().item() for name, metric in self._model.metrics['val'].items()}

        return (
            train_metrics['f1'],
            train_metrics,
            val_metrics['f1'],
            val_metrics,
        )
