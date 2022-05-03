from types import MappingProxyType

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

TFIDF_GRID = MappingProxyType({
    'input': ('content',),
    'encoding': ('utf-8',),
    'decode_error': ('ignore',),
    'ngram_range': (
        (1, 1),
        (2, 2),
        (3, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ),
    'max_df': np.linspace(
        start=0,
        stop=1,
        num=25,
    ),
    'min_df': np.linspace(
        start=0,
        stop=1,
        num=25,
    ),
    'max_features': (
        100,
        150,
        200,
        250,
        300,
    ),
    'binary': (
        False,
        True,
    ),
    'norm': (
        'l1',
        'l2',
    ),
    'use_idf': (
        False,
        True,
    ),
    'smooth_idf': (
        False,
        True,
    ),
    'sublinear_tf': (
        False,
        True,
    ),
})

CLASSIFIERS = (
    (
        LogisticRegression,
        MappingProxyType({
            'penalty': (
                'l1',
                'l2',
                'elasticnet',
                'none',
            ),
            'dual': (True,),
            'C': (
                0.1,
                1,
                10,
                100,
                1000,
            ),
            'solver': (
                'newton-cg',
                'lbfgs',
                'liblinear',
                'sag',
                'saga',
            ),
            'l1_ratio': np.linspace(
                start=0.1,
                stop=0.9,
                num=10,
            ),
            'max_iter': (1000,),
            'n_jobs': (-1,),
        }),
    ),
    (
        KNeighborsClassifier,
        MappingProxyType({
            'n_neighbors': (
                3,
                5,
                7,
                9,
            ),
            'weights': (
                'uniform',
                'distance',
            ),
            'leaf_size': (
                10,
                20,
                30,
                40,
                50,
            ),
            'n_jobs': (-1,),
        }),
    ),
    (
        RandomForestClassifier,
        MappingProxyType({
            'n_estimators': (
                10,
                50,
                100,
                200,
                300,
            ),
            'criterion': (
                'gini',
                'entropy',
            ),
            'n_jobs': (-1,),
        }),
    ),
    (
        GaussianNB,
        MappingProxyType({
            'var_smoothing': (
                1e-3,
                1e-6,
                1e-9,
                1e-12,
            ),
        }),
    )
)
