from types import MappingProxyType

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

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
    ),
)
