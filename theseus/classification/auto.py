from pathlib import Path
from typing import Optional

from theseus._mixin import AutoEstimatorMixin
from theseus.abc.auto_estimator import AutoEstimator
from theseus.accelerator import Accelerator
from theseus.classification.bert_classifier import BertClassifier
from theseus.classification.embeddings import (
    FastTextClassifier,
    SentenceBertClassifier,
    TfIdfClassifier,
)
from theseus.dataset.balancing.balancer import DatasetBalancer
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import DeviceError
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger
from theseus.plotting.classification import plot_class_distribution

_logger = setup_logger(__name__)


class AutoClassifier(AutoEstimator, AutoEstimatorMixin):
    def __init__(
        self,
        out_dir: Path,
        accelerator: Accelerator,
        target_lang: Optional[LanguageCode] = None,
        ignore_imbalance: bool = False,
    ) -> None:
        super().__init__(
            out_dir,
            accelerator,
            target_lang,
        )

        self._ignore_imbalance = ignore_imbalance

    def fit(
        self,
        dataset: TextDataset,
    ) -> None:
        plot_class_distribution(
            dataset.labels,
            self._out_dir / 'class_distribution.png',
        )

        _logger.info('detecting_language')
        self._target_lang = self._detect_lang(
            self._target_lang,
            dataset.texts,
        )

        _logger.info('balancing dataset')
        balancer = DatasetBalancer(
            self._target_lang,
            self._ignore_imbalance,
        )
        dataset = balancer(dataset)

        # tfidf
        _logger.info('trying TF-IDF classification')
        tf_idf_path = self._out_dir / 'tf-idf'
        tf_idf_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        clf = TfIdfClassifier(
            self._target_lang,
            tf_idf_path,
        )
        score = clf.fit(dataset)
        _logger.info(f'best F1 score with TF-IDF: {score:.4f}')

        # fasttext
        ft_path = self._out_dir / 'ft'
        ft_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        clf = FastTextClassifier(
            self._target_lang,
            ft_path,
        )
        score = clf.fit(dataset)
        _logger.info(f'best F1 score with fastText embeddings: {score:.4f}')

        try:
            device = self._accelerator.select_single_gpu()
        except DeviceError:
            _logger.error('no suitable GPU was found, skipping SentenceBERT & BERT classifier')
        else:
            # sentence_bert
            sbert_path = self._out_dir / 'sbert'
            sbert_path.mkdir(
                parents=True,
                exist_ok=True,
            )
            clf = SentenceBertClassifier(
                self._target_lang,
                sbert_path,
                device,
            )
            score = clf.fit(dataset)
            _logger.info(f'best F1 score with SentenceBERT embeddings: {score:.4f}')

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
            score = clf.fit(dataset)
            _logger.info(f'best F1 score with BERT classifier: {score:.4f}')
