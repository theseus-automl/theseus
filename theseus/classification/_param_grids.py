from types import MappingProxyType

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

CLASSIFIERS = (
    (
        LogisticRegression,
        MappingProxyType({
            'penalty': (
                'l1',
                'l2',
                'elasticnet',
            ),
            'dual': (True,),
            'C': (
                0.1,
                1,
                10,
                100,
                1000,
            ),
            'l1_ratio': np.linspace(
                start=0.1,
                stop=0.9,
                num=5,
            ),
            'max_iter': (500,),
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
            'criterion': ('gini',),
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
