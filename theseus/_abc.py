import warnings
from abc import ABC
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Union,
)

import joblib
from sklearn.exceptions import (
    ConvergenceWarning,
    FitFailedWarning,
)
from sklearn.metrics._scorer import _PredictScorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger
from theseus.plotting.classification import (
    plot_gs_result,
    plot_metrics,
)
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

    def fit(
        self,
        dataset: TextDataset,
    ) -> float:
        result = []

        if callable(self._models):
            self._models = self._models(len(dataset))

        for clf, clf_param_grid in self._models:
            _logger.info(f'trying {clf.__name__}')

            if dataset.class_weights is None:
                _logger.info('no class weights will be used')
            else:
                _logger.info('using class weights for classification')
                clf_param_grid['class_weight'] = dataset.class_weights

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
                clf_param_grid,
                'clf',
            ))

            grid = GridSearchCV(
                pipeline,
                dict(param_grid),
                scoring=self._scoring,
                refit=self._refit,
                error_score=0,  # to avoid forbidden combinations
                return_train_score=True,
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

        return result[0]['best_score']

    @staticmethod
    def _add_param_grid_prefix(
        param_grid: Union[dict, MappingProxyType],
        prefix: str,
    ) -> Dict[str, Any]:
        prefixed = {}

        for name, options in param_grid.items():
            prefixed[f'{prefix}__{name}'] = options

        return prefixed
