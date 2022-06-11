from theseus.abc.auto_estimator import AutoEstimator
from theseus.clustering.embeddings import (
    FastTextClusterer,
    SentenceBertClusterer,
    TfIdfClusterer,
)
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import DeviceError
from theseus.log import setup_logger

_logger = setup_logger(__name__)


class AutoClusterer(AutoEstimator):
    def fit(
        self,
        dataset: TextDataset,
    ) -> None:
        _logger.info('detecting_language')
        self._detect_lang(dataset.texts)

        # tfidf
        if self._use_tf_idf:
            self._fit_tf_idf(dataset)
        else:
            _logger.warning('skipping TF-IDF')

        # fasttext
        if self._use_fasttext:
            self._fit_fast_text(dataset)
        else:
            _logger.warning('skipping fastText')

        if self._use_bert:
            self._fit_bert(dataset)
        else:
            _logger.warning('skipping SBERT')

    def _fit_tf_idf(
        self,
        dataset: TextDataset,
    ) -> None:
        _logger.info('trying TF-IDF clustering')
        tf_idf_path = self._out_dir / 'tf-idf'
        tf_idf_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        clf = TfIdfClusterer(
            self._target_lang,
            tf_idf_path,
            n_jobs=self._tf_idf_n_jobs,
            n_iter=self._tf_idf_n_iter,
        )
        score, metrics = clf.fit(dataset)
        self._log_score(
            _logger,
            score,
            metrics,
            'TF-IDF',
            'silhouette',
        )

    def _fit_fast_text(
        self,
        dataset: TextDataset,
    ) -> None:
        _logger.info('trying fastText clustering')
        ft_path = self._out_dir / 'ft'
        ft_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        clf = FastTextClusterer(
            self._target_lang,
            ft_path,
            n_jobs=self._fast_text_n_jobs,
            n_iter=self._fast_text_n_iter,
        )
        score, metrics = clf.fit(dataset)
        self._log_score(
            _logger,
            score,
            metrics,
            'fastText embeddings',
            'silhouette',
        )

    def _fit_bert(
        self,
        dataset: TextDataset,
    ) -> None:
        try:
            device = self._accelerator.select_single_gpu()
        except DeviceError:
            _logger.error('no suitable GPU was found, skipping SentenceBERT')
        else:
            # sentence_bert
            _logger.info('trying SentenceBERT clustering')
            sbert_path = self._out_dir / 'sbert'
            sbert_path.mkdir(
                parents=True,
                exist_ok=True,
            )
            clf = SentenceBertClusterer(
                self._target_lang,
                sbert_path,
                device,
                n_jobs=self._sbert_n_jobs,
                n_iter=self._sbert_n_iter,
            )
            score, metrics = clf.fit(dataset)
            self._log_score(
                _logger,
                score,
                metrics,
                'SentenceBERT embeddings',
                'silhouette',
            )
