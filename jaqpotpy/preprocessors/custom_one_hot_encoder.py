from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import warnings
import pandas as pd


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """A custom one-hot encoder for pandas DataFrames.

    Parameters
    ----------
    categorical_columns : list of str, optional
        List of column names to be treated as categorical and to be one-hot encoded.

    """

    _CATEGORY_SEPARATOR = "."
    _PROTECTED_KEYWORDS = ["SMILES"]

    def __init__(self, categorical_columns=[]):
        self._sklearn_is_fitted = False
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        """Fit the encoder to the DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame to fit.
        y : None
            Ignored. This parameter exists for compatibility with scikit-learn's interface.

        Returns
        -------
        self : CustomOneHotEncoder
            Fitted encoder.

        Raises
        ------
        ValueError
            If `X` is not a pandas DataFrame.
            If a column name contains the protected separator.
            If a column indicated as `categorical`, contains float values.

        """
        if not isinstance(X, pd.DataFrame):
            msg = "X must be of type pd.DataFrame"
            raise ValueError(msg)

        self.X_columns_ = list(X.columns)
        self.categories_ = {}
        self.new_columns_ = []

        for keyword in self._PROTECTED_KEYWORDS:
            if any(col.upper() == keyword.upper() for col in self.X_columns_):
                msg = (
                    f"The {keyword.upper()} keyword is in a protected namespace and should not be used. "
                    "This check is case-insensitive."
                )
                raise ValueError(msg)

        if any(self._CATEGORY_SEPARATOR in col for col in X.columns):
            msg = f"Column names of input Dataframe must not contain '{self._CATEGORY_SEPARATOR}''"
            raise ValueError(msg)

        categorical_compatible_columns = set(
            X.select_dtypes(include=["object"]).columns
        )
        if categorical_compatible_columns - set(self.categorical_columns):
            for col in categorical_compatible_columns:
                msg = f"Column '{col}' has non-numerical values and might be suitable for one-hot-encoding."
                warnings.warn(msg)

        X_columns_set = set(self.X_columns_)
        for col in self.categorical_columns:
            if col not in X_columns_set:
                msg = f"Column '{col}' is not in the DataFrame"
                raise ValueError(f"Column '{col}' is not in the DataFrame")

            if X[col].dtype == float:
                msg = f"Column '{col}' contains float values and cannot be treated as categorical."
                raise ValueError(msg)

            unique_values = X[col].unique().tolist()
            if set(unique_values) == {0, 1}:
                msg = f"Column '{col}' has only two unique values (0 and 1) and could be treated as a binary variable."
                warnings.warn(msg)
            self.categories_[col] = unique_values

        for col in X.columns:
            if col in self.categories_:
                self.new_columns_.extend(
                    [
                        f"{col}{self._CATEGORY_SEPARATOR}{val}"
                        for val in self.categories_[col]
                    ]
                )
            else:
                self.new_columns_.append(col)

        self._sklearn_is_fitted = True
        return self

    def transform(self, X):
        """Transform the DataFrame using the fitted encoder.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame to transform.

        Returns
        -------
        encoded_df : pandas.DataFrame
            The transformed DataFrame with one-hot encoded columns.

        Raises
        ------
        NotFittedError
            If the encoder is not fitted.
        ValueError
            If `X` is not a pandas DataFrame.
            If `X` does not have the exact same columns as the DataFrame used for fitting.

        """
        if not self.__sklearn_is_fitted__():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        if not isinstance(X, pd.DataFrame):
            msg = "X must be of type pd.DataFrame"
            raise ValueError(msg)

        if list(X.columns) != self.X_columns_:
            msg = "X must have the exact same columns used for fitting."
            raise ValueError(msg)

        encoded_df = X.copy()

        for col in self.categorical_columns:
            if col in encoded_df.columns:
                unique_values = self.categories_[col]

                unexpected_values = set(encoded_df[col].unique()) - set(unique_values)
                if unexpected_values:
                    msg = f"Column '{col}' contains unexpected values: {unexpected_values}."
                    warnings.warn(msg, UserWarning, stacklevel=2)

                for val in unique_values:
                    encoded_df[f"{col}{self._CATEGORY_SEPARATOR}{val}"] = (
                        encoded_df[col] == val
                    ).astype(int)
                encoded_df.drop(col, axis=1, inplace=True)

        encoded_df = encoded_df[self.new_columns_]

        return encoded_df

    def __sklearn_is_fitted__(self):
        """Check if the estimator is fitted.

        Returns
        -------
        bool
            True if the estimator is fitted, False otherwise.

        """
        return self._sklearn_is_fitted
