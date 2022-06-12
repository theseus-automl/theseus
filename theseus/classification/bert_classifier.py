import pickle
from pathlib import Path
from types import MappingProxyType

import pytorch_lightning as pl

from theseus.accelerator import Accelerator
from theseus.classification.models.bert import BertForClassification
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.plotting.display import save_fig

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
    ) -> float:
        self._model.set_data(dataset)

        logger = pl.loggers.TensorBoardLogger(
            save_dir=self._out_dir / 'logs',
            name='BERT Classifier',
        )
        trainer = pl.Trainer(
            logger=logger,
            # auto_scale_batch_size='power',
            # auto_lr_find=True,
            max_epochs=_MAX_EPOCHS,
            deterministic=True,
            **self._accelerator_params,
        )

        trainer.tuner.scale_batch_size(self._model)
        lr_finder = trainer.tuner.lr_find(
            self._model,
            min_lr=1e-8,
            max_lr=1e-5,
        )
        save_fig(
            self._out_dir / 'lr_tuner.png',
            False,
            lr_finder.plot(suggest=True),
        )

        print(self._model.batch_size)

        trainer.fit(self._model)

        try:
            with open(self._out_dir / 'metrics', 'wb') as f:
                pickle.dump(
                    self._model.metrics,
                    f,
                )
        except Exception:
            print('unable to pickle metrics')

        return self._model.metrics['val']['f1'].compute().item()
