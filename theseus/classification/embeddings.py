import warnings
from abc import ABC
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Optional,
)

import joblib
from sklearn.exceptions import (
    ConvergenceWarning,
    FitFailedWarning,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from theseus.classification._metrics import CLASSIFICATION_METRICS
from theseus.classification._param_grids import CLASSIFIERS
from theseus.classification._utils import add_param_grid_prefix
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger
from theseus.plotting.classification import plot_gs_result, plot_metrics
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


class EmbeddingsClassifier(ABC):
    _out_dir = ExistingDir()

    def __init__(
        self,
        target_lang: LanguageCode,
        out_dir: Path,
        embedder: Any,
        param_grid: Optional[Dict[str, Any]] = None,
        supported_languages: Optional[MappingProxyType] = None,
    ) -> None:
        if supported_languages is not None and target_lang not in supported_languages:
            raise UnsupportedLanguageError(f'{self.__class__.__name__} is unavailable for {target_lang}')

        self._target_lang = target_lang
        self._out_dir = out_dir
        self._embedder = embedder
        self._emb_param_grid = {} if param_grid is None else param_grid

    def fit(
        self,
        dataset: TextDataset,
    ) -> float:
        result = []

        for clf, clf_param_grid in CLASSIFIERS:
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
            param_grid = add_param_grid_prefix(
                self._emb_param_grid,
                'emb',
            )
            param_grid.update(add_param_grid_prefix(
                clf_param_grid,
                'clf',
            ))

            grid = GridSearchCV(
                pipeline,
                dict(param_grid),
                scoring=dict(CLASSIFICATION_METRICS),
                refit='f1',
                error_score=0,  # to avoid forbidden combinations
                return_train_score=True,
            )
            grid.fit(
                dataset.texts,
                dataset.labels,
            )

            plot_gs_result(
                grid.cv_results_,
                dict(CLASSIFICATION_METRICS),
                self._out_dir,
            )
            plot_metrics(
                grid.cv_results_,
                self._out_dir,
            )

            result.append(
                {
                    'classifier': grid.best_estimator_,
                    'best_score': grid.best_score_,
                },
            )

        result.sort(
            key=lambda item: item['best_score'],
            reverse=True,
        )

        raw_model_path = self._out_dir / 'raw_clf.pkl'
        _logger.info(f'saving model to {raw_model_path.resolve()}')
        joblib.dump(
            result[0]['classifier'],
            raw_model_path,
        )

        return result[0]['best_score']
