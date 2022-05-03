import warnings
from abc import (
    ABC,
    abstractmethod,
)
from pathlib import Path
from types import MappingProxyType
from typing import (
    List,
    Optional,
)

import joblib
import numpy as np
import torch
from skl2onnx import to_onnx
from sklearn.base import BaseEstimator
from sklearn.exceptions import (
    ConvergenceWarning,
    FitFailedWarning,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV

from theseus.classification._param_grids import CLASSIFIERS
from theseus.dataset.text_dataset import TextDataset
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger
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
        supported_languages: Optional[MappingProxyType] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if supported_languages is not None and target_lang not in supported_languages:
            raise UnsupportedLanguageError(f'{self.__class__.__name__} is unavailable for {target_lang}')

        self._target_lang = target_lang
        self._out_dir = out_dir
        self._device = device

    def fit(
        self,
        dataset: TextDataset,
    ) -> float:
        embeddings = self._embed(dataset.texts)
        embeddings_path = self._out_dir / 'embeddings.npy'
        _logger.info(f'saving embeddings to {embeddings_path.resolve()}')
        np.save(
            embeddings_path,
            embeddings,
        )

        result = []

        for clf, param_grid in CLASSIFIERS:
            _logger.info(f'trying {clf.__name__}')

            if dataset.class_weights is None:
                _logger.info('no class weights will be used')
            else:
                _logger.info('using class weights for classification')
                param_grid['class_weight'] = dataset.class_weights

            grid = GridSearchCV(
                clf(),
                dict(param_grid),
                n_jobs=-1,
                scoring={
                    'accuracy': make_scorer(accuracy_score),
                    'f1': make_scorer(f1_score),
                    'precision': make_scorer(precision_score),
                    'recall': make_scorer(recall_score),
                },
                refit='f1',
                error_score=0,  # to avoid forbidden combinations
            )

            grid.fit(
                embeddings,
                dataset.labels,
            )

            result.append(
                {
                    'classifier': grid.best_estimator_,
                    'best_score': grid.best_score_,
                }
            )

        result.sort(
            key=lambda item: item['best_score'],
            reverse=True,
        )

        raw_model_path = self._out_dir / 'raw_clf.pkl'
        _logger.info(f'saving raw model to {raw_model_path.resolve()}')
        joblib.dump(
            result[0]['classifier'],
            raw_model_path,
        )

        self._to_onnx(
            result[0]['classifier'],
            embeddings[0],
            self._out_dir / 'clf.onnx',
        )

        return result[0]['best_score']

    @abstractmethod
    def _embed(
        self,
        texts: List[str],
    ) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _to_onnx(
        model: BaseEstimator,
        sample: np.ndarray,
        path: Path,
    ) -> None:
        _logger.info(f'converting to ONNX...')
        onx = to_onnx(
            model,
            sample.astype(np.float32),
        )

        _logger.info(f'saving ONNX model to {path.resolve()}')

        with open(path, 'wb') as f:
            f.write(onx.SerializeToString())
