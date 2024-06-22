"""
Author: Ioannis Pitoskas
Contact: jpitoskas@gmail.com
"""

from .custom_one_hot_encoder import CustomOneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class CustomOneHotTransformerPipeline(BaseEstimator, TransformerMixin):
    """
    A custom pipeline that starts with a CustomOneHotEncoder and then applies a series of user-defined steps.

    Parameters
    ----------
    categorical_columns : list of str, optional
        List of column names to be treated as categorical and to be one-hot encoded.
    steps : list of tuples, optional
        List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are passed.
    """

    def __init__(self, categorical_columns=[], steps=None):
        self._sklearn_is_fitted = False
        self.categorical_columns = categorical_columns
        self.steps = steps if steps is not None else []

        for estimator_name, estimator in self.steps:
            if not hasattr(estimator, 'transform'):
                msg = (
                    f"Estimator '{estimator_name}' of type {type(estimator).__name__} does not have a 'transform' method. "
                    "All pipeline steps must implement the 'transform' method."
                    )
                raise ValueError(msg)
            

        self.one_hot_encoder = CustomOneHotEncoder(categorical_columns=categorical_columns)
        self.pipeline = Pipeline([('one_hot_encoder', self.one_hot_encoder)] + self.steps)

    def fit(self, X, y=None):
        """
        Fit the pipeline.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame to fit.
        y : array-like, optional
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : CustomPipeline
            Fitted pipeline.
        """
        self.pipeline.fit(X, y)
        self.X_columns_ = list(self.one_hot_encoder.X_columns_)
        self.categories_ = dict(self.one_hot_encoder.categories_)
        self.new_columns_ = list(self.one_hot_encoder.new_columns_)
        self._sklearn_is_fitted = True
        return self

    def transform(self, X):
        """
        Transform the DataFrame using the fitted pipeline.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame to transform.

        Returns
        -------
        transformed_X : pandas.DataFrame
            The transformed DataFrame.
        """

        if not self.__sklearn_is_fitted__():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})
        
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame to fit.
        y : array-like, optional
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        transformed_X : pandas.DataFrame
            The transformed DataFrame.
        """
        return self.pipeline.fit_transform(X, y)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return self.pipeline.get_params(deep=deep)
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : CustomPipeline
            Estimator instance.
        """
        self.pipeline.set_params(**params)
        return self
    
    def __sklearn_is_fitted__(self):
        """
        Check if the estimator is fitted.

        Returns
        -------
        bool
            True if the estimator is fitted, False otherwise.
        """
        return self._sklearn_is_fitted