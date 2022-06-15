from pathlib import Path
from timeit import default_timer as timer
from typing import Optional

import torch

from theseus.abc.auto_estimator import AutoEstimator
from theseus.accelerator import Accelerator
from theseus.classification.bert_classifier import BertClassifier
from theseus.classification.embeddings import (
    FastTextClassifier,
    TfIdfClassifier,
)
from theseus.dataset.balancing.balancer import DatasetBalancer
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import DeviceError
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger
from theseus.plotting.classification import plot_class_distribution

_logger = setup_logger(__name__)


class AutoClassifier(AutoEstimator):
    def __init__(
        self,
        out_dir: Path,
        accelerator: Accelerator,
        target_lang: Optional[LanguageCode] = None,
        *,
        ignore_imbalance: bool = False,
        use_tf_idf: bool = True,
        use_fasttext: bool = True,
        use_bert: bool = True,
        tf_idf_n_jobs: int = -1,
        fast_text_n_jobs: int = -1,
        tf_idf_n_iter: int = 5,
        fast_text_n_iter: int = 10,
    ) -> None:
        super().__init__(
            out_dir,
            accelerator,
            target_lang,
            use_tf_idf=use_tf_idf,
            use_fasttext=use_fasttext,
            use_bert=use_bert,
            tf_idf_n_jobs=tf_idf_n_jobs,
            fast_text_n_jobs=fast_text_n_jobs,
            tf_idf_n_iter=tf_idf_n_iter,
            fast_text_n_iter=fast_text_n_iter,
        )

        self._ignore_imbalance = ignore_imbalance

    def fit(
        self,
        dataset: TextDataset,
    ) -> None:
        plot_class_distribution(
            dataset.le.inverse_transform(dataset.labels),
            self._out_dir / 'class_distribution.png',
        )

        _logger.info('detecting_language')
        self._detect_lang(dataset.texts)

        _logger.info('balancing dataset')
        balancer = DatasetBalancer(
            self._target_lang,
            self._ignore_imbalance,
            device=self._pick_device_for_balancing(),
        )
        start = timer()
        dataset = balancer(dataset)
        _logger.info(f'dataset balancing took: {timer() - start} seconds')

        # tfidf
        if self._use_tf_idf:
            self._fit_tf_idf(dataset)
        else:
            _logger.warning('skipping TF-IDF')

        # fasttext
        if self._use_fasttext:
            self._fit_fasttext(dataset)
        else:
            _logger.warning('skipping fastText')

        # bert
        if self._use_bert:
            self._fit_bert(dataset)
        else:
            _logger.warning('skipping BERT')

    def _fit_tf_idf(
        self,
        dataset: TextDataset,
    ) -> None:
        _logger.info('trying TF-IDF classification')
        tf_idf_path = self._out_dir / 'tf-idf'
        tf_idf_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        clf = TfIdfClassifier(
            self._target_lang,
            tf_idf_path,
            n_jobs=self._tf_idf_n_jobs,
            n_iter=self._tf_idf_n_iter,
        )
        start = timer()
        score, metrics = clf.fit(dataset)
        _logger.info(f'TF-IDF took: {timer() - start} seconds')

        self._log_score(
            _logger,
            score,
            metrics,
            'TF-IDF',
            'F1',
        )

    def _fit_fasttext(
        self,
        dataset: TextDataset,
    ) -> None:
        ft_path = self._out_dir / 'ft'
        ft_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        clf = FastTextClassifier(
            self._target_lang,
            ft_path,
            n_jobs=self._fast_text_n_jobs,
            n_iter=self._fast_text_n_iter,
        )
        start = timer()
        score, metrics = clf.fit(dataset)
        _logger.info(f'fastText took: {timer() - start} seconds')

        self._log_score(
            _logger,
            score,
            metrics,
            'fastText',
            'F1',
        )

    def _fit_bert(
        self,
        dataset: TextDataset,
    ) -> None:
        try:
            self._accelerator.select_single_gpu()
        except DeviceError:
            _logger.error('no suitable GPU was found, skipping BERT classifier')
        else:
            # bert
            bert_path = self._out_dir / 'bert'
            bert_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            clf = BertClassifier(
                self._target_lang,
                dataset.num_labels,
                bert_path,
                self._accelerator,
            )
            start = timer()
            train_score, train_metrics, val_score, val_metrics = clf.fit(dataset)
            _logger.info(f'BERT took: {timer() - start} seconds')

            _logger.info('TRAIN METRICS')
            self._log_score(
                _logger,
                train_score,
                train_metrics,
                'BERT classifier',
                'F1',
            )

            _logger.info('VAL METRICS')
            self._log_score(
                _logger,
                val_score,
                val_metrics,
                'BERT classifier',
                'F1',
            )

    def _pick_device_for_balancing(
        self,
    ) -> Optional[torch.device]:
        try:
            return self._accelerator.select_single_gpu()
        except DeviceError:
            return None
