from pathlib import Path
from types import MappingProxyType

import pytorch_lightning as pl
import torch
from transformers import (
    BatchEncoding,
    BertTokenizer,
)

from theseus.accelerator import Accelerator
from theseus.classification.models.bert import BertForClassification
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode

_SUPPORTED_LANGS = MappingProxyType({
    LanguageCode.RUSSIAN: 'DeepPavlov/rubert-base-cased',
})

_MAX_EPOCHS = 15


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
    ) -> None:
        self._model.set_data(dataset)

        logger = pl.loggers.TensorBoardLogger(
            save_dir=self._out_dir / 'logs',
            name='BERT Classifier',
        )
        trainer = pl.Trainer(  # TODO: accelerator
            logger=logger,
            auto_scale_batch_size='power',
            auto_lr_find=True,
            max_epochs=_MAX_EPOCHS,
            deterministic=True,
            **self._accelerator_params,
        )
        trainer.tune(self._model)
        trainer.fit(self._model)

    @staticmethod
    def collate_fn(
        input_data,
        tokenizer: BertTokenizer,
    ) -> BatchEncoding:
        texts, labels = zip(*input_data)
        labels = torch.LongTensor(labels)

        inputs = tokenizer(
            texts,
            return_tensors='pt',
            padding='longest',
            max_length=256,
            truncation=True,
        )
        inputs['Class'] = labels

        return inputs
