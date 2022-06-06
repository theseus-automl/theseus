from types import MappingProxyType

from scipy.stats import uniform
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
            'C': uniform(
                loc=0.1,
                scale=1000 - 0.1,
            ),
            'l1_ratio': uniform(
                loc=0.1,
                scale=0.9 - 0.1,
            ),
            'max_iter': (500,),
            'n_jobs': (-1,),
        }),
    ),
    (
        RandomForestClassifier,
        MappingProxyType({
            'n_estimators': uniform(
                loc=10,
                scale=300 - 10,
            ),
            'criterion': ('gini',),
            'n_jobs': (-1,),
        }),
    ),
    (
        GaussianNB,
        MappingProxyType({
            'var_smoothing': uniform(
                loc=1e-3,
                scale=1e-12 - 1e-3,
            ),
        }),
    ),
)
