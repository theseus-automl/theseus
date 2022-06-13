import warnings
from abc import ABC
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
)

import joblib
import numpy as np
from sklearn.exceptions import (
    ConvergenceWarning,
    FitFailedWarning,
)
from sklearn.metrics._scorer import _PredictScorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from theseus.cv import make_split
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger
from theseus.plotting.classification import (
    plot_gs_result,
    plot_metrics,
)
from theseus.plotting.clustering import plot_clustering_results
from theseus.utils import get_args_names
from theseus.validators import ExistingDir

_logger = setup_logger(__name__)

warnings.simplefilter(
    'ignore',
    category=UserWarning,
)
warnings.simplefilter(
    'ignore',
    category=ConvergenceWarning,
)
warnings.simplefilter(
    'ignore',
    category=FitFailedWarning,
)


class EmbeddingsEstimator(ABC):
    _out_dir = ExistingDir()

    def __init__(
        self,
        target_lang: LanguageCode,
        out_dir: Path,
        embedder: Any,
        models: Union[tuple, Callable],
        scoring: Dict[str, _PredictScorer],
        refit: str,
        embedder_param_grid: Optional[Dict[str, Any]] = None,
        supported_languages: Optional[MappingProxyType] = None,
        n_jobs: int = -1,
        n_iter: int = 10,
    ) -> None:
        if supported_languages is not None and target_lang not in supported_languages:
            raise UnsupportedLanguageError(f'{self.__class__.__name__} is unavailable for {target_lang}')

        if refit not in scoring:
            raise ValueError(f'{refit} metric was chosen as refit, but scoring does not include it')

        self._target_lang = target_lang
        self._out_dir = out_dir
        self._embedder = embedder
        self._models = models
        self._scoring = scoring
        self._refit = refit
        self._emb_param_grid = {} if embedder_param_grid is None else embedder_param_grid
        self._n_jobs = n_jobs
        self._n_iter = n_iter

    def fit(
        self,
        dataset: TextDataset,
    ) -> Tuple[float, Dict[str, float]]:
        result = []

        if callable(self._models):
            self._models = self._models(len(dataset))

        # if self._n_iter < len(self._models):
        #     raise ValueError('n_iter can not be smaller than number of classifiers')
        #
        # if self._n_iter % len(self._models) != 0:
        #     raise ValueError(f'n_iter must be divisible by {len(self._models)}')
        #
        # self._n_iter %= len(self._models)

        for clf, clf_param_grid in self._models:
            _clf_param_grid = dict(clf_param_grid)
            _logger.info(f'trying {clf.__name__}')

            if dataset.class_weights is None:
                if dataset.labels is not None:
                    _logger.info('no class weights will be used')
            elif 'class_weight' in get_args_names(clf.__init__):
                _logger.info('using class weights for classification')
                _clf_param_grid['class_weight'] = (dataset.class_weights,)  # make tuple for random search

            pipeline = Pipeline(
                [
                    (
                        'emb',
                        self._embedder,
                    ),
                    (
                        'clf',
                        clf(),
                    ),
                ],
            )
            param_grid = self._add_param_grid_prefix(
                self._emb_param_grid,
                'emb',
            )
            param_grid.update(self._add_param_grid_prefix(
                _clf_param_grid,
                'clf',
            ))

            grid = RandomizedSearchCV(
                pipeline,
                param_grid,
                scoring=self._scoring,
                refit=self._refit,
                cv=make_split(dataset),
                error_score=-100,  # to avoid forbidden combinations
                return_train_score=True,
                verbose=2,
                n_jobs=self._n_jobs,
                n_iter=self._n_iter,
            )
            grid.fit(
                dataset.texts,
                dataset.labels,
            )

            plot_gs_result(
                grid.cv_results_,
                self._scoring,
                self._out_dir,
            )
            plot_metrics(
                grid.cv_results_,
                self._out_dir,
            )

            result.append(
                {
                    'estimator': grid.best_estimator_,
                    'best_score': grid.best_score_,
                    'metrics': self._collect_metrics(grid.cv_results_),
                    'cv_results': grid.cv_results_,
                },
            )

        result.sort(
            key=lambda item: item['best_score'],
            reverse=True,
        )

        raw_model_path = self._out_dir / 'raw_estimator.pkl'
        _logger.info(f'saving model to {raw_model_path.resolve()}')
        joblib.dump(
            result[0]['estimator'],
            raw_model_path,
        )

        cv_results_path = self._out_dir / 'cv_results.pkl'
        _logger.info(f'saving CV results to {cv_results_path.resolve()}')
        joblib.dump(
            result,
            cv_results_path,
        )

        if dataset.labels is None:
            plot_clustering_results(
                result[0]['estimator']['emb'].transform(dataset.texts),
                result[0]['estimator']['clf'].labels_,
                self._out_dir / 'result.png',
            )

        return (
            result[0]['best_score'],
            result[0]['metrics'],
        )

    def _collect_metrics(
        self,
        gs_result: Dict[str, Any],
    ) -> Dict[str, float]:
        best_index = np.nonzero(gs_result[f'rank_test_{self._refit}'] == 1)[0][0]
        metrics = {}

        for metric in self._scoring:
            metrics[metric] = gs_result[f'mean_test_{metric}'][best_index]

        return metrics

    @staticmethod
    def _add_param_grid_prefix(
        param_grid: Union[dict, MappingProxyType],
        prefix: str,
    ) -> Dict[str, Any]:
        prefixed = {}

        for name, options in param_grid.items():
            prefixed[f'{prefix}__{name}'] = options

        return prefixed
