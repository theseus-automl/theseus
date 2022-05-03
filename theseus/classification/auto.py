from pathlib import Path

from theseus.accelerator import Accelerator
from theseus.classification.bert_classifier import BertClassifier
from theseus.classification.fast_text import FastTextClassifier
from theseus.classification.sentence_bert import SentenceBertClassifier
from theseus.classification.tf_idf import TfIdfClassifier
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import DeviceError
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger
from theseus.plotting.classification import plot_class_distribution

_logger = setup_logger(__name__)


class AutoClassifier:
    def __init__(
        self,
        target_lang: LanguageCode,
        out_dir: Path,
        accelerator: Accelerator,
    ) -> None:
        self._target_lang = target_lang
        self._out_dir = out_dir
        self._accelerator = accelerator

        self._out_dir.mkdir(
            exist_ok=True,
            parents=True,
        )

    def fit(
        self,
        dataset: TextDataset,
    ) -> None:
        plot_class_distribution(
            dataset.labels,
            self._out_dir / 'class_distribution.png',
        )

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

        # sentence_bert
        try:
            device = self._accelerator.select_single_gpu()
        except DeviceError:
            _logger.error('no suitable GPU was found, skipping SentenceBERT and BERT classifier')
            device = None

        if device is not None:
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
            clf.fit(dataset)

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
            clf.fit(dataset)
