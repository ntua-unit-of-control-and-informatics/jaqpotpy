from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.info = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        return np.exp(X)
