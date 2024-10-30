"""Dataset classes for molecular modelling"""

from typing import Iterable, Optional, Any, List
import copy
import pandas as pd
import sklearn.feature_selection as skfs
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from jaqpotpy.datasets.dataset_base import BaseDataset


class JaqpotpyDataset(BaseDataset):
    """A dataset class that can also support molecular descriptors. This class is
    designed to handle datasets that include molecular structures represented by
    SMILES strings and use molecular featurizers to generate features from these
    structures. The class can also support ordinary datasets.

    Attributes
    ----------
        df:
        path:
        y_cols:
        x_cols:
        smiles_col : The column containing SMILES strings.
        x_cols : The SMILES strings extracted from the DataFrame.

        featurizer: The featurizer used to generate molecular features. It can be one of the supported
                    featurizers or a list of supported featurizers
        task : All feature columns after featurization.

    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        path: Optional[str] = None,
        y_cols: Iterable[str] = None,
        x_cols: Optional[Iterable[str]] = None,
        smiles_cols: Optional[Iterable[str]] = None,
        featurizer: Optional[List[MolecularFeaturizer] or MolecularFeaturizer] = None,
        task: str = None,
    ) -> None:
        if not (
            isinstance(smiles_cols, str)
            or (
                isinstance(smiles_cols, list)
                and all(isinstance(item, str) for item in smiles_cols)
            )
            or (isinstance(smiles_cols, list) and len(smiles_cols) == 0)
            or (smiles_cols is None)
        ):
            raise TypeError(
                "smiles_cols should be either a string, an empty list"
                "a list of strings, or None"
            )

        if (smiles_cols is not None) and (featurizer is None):
            raise TypeError(
                "Cannot estimate SMILES descriptors without a featurizer."
                "Please provide a featurizer"
            )

        # Find the length of each provided column name vector and put everything in lists
        if isinstance(smiles_cols, str):
            self.smiles_cols = [smiles_cols]
            self.smiles_cols_len = 1
        elif isinstance(smiles_cols, list):
            self.smiles_cols = smiles_cols
            self.smiles_cols_len = len(smiles_cols)
        elif smiles_cols is None:
            self.smiles_cols = []
            self.smiles_cols_len = 0

        if featurizer is not None:
            if isinstance(featurizer, list):
                for individual_featurizer in featurizer:
                    if not isinstance(individual_featurizer, MolecularFeaturizer):
                        raise TypeError(
                            "Each featurizer in the featurizer list should be a MolecularFeaturizer instance."
                        )
            elif isinstance(featurizer, MolecularFeaturizer):
                featurizer = [featurizer]
            else:
                raise TypeError(
                    "featurizer should be a list containing MolecularFeaturizer instances."
                )

        super().__init__(df=df, path=path, y_cols=y_cols, x_cols=x_cols, task=task)

        self._validate_column_overlap(self.smiles_cols, self.x_cols, self.y_cols)
        self._validate_column_names(self.smiles_cols, "smiles_cols")
        self._validate_column_names(self.x_cols, "x_cols")
        self._validate_column_names(self.y_cols, "y_cols")
        self._validate_column_space()

        self.init_df = self.df
        self.featurizer = featurizer
        # If featurizer is provided and it's for training, we need to copy the attributes
        if self.featurizer:
            self.featurizers_attributes = {}
            for featurizer_i in self.featurizer:
                self.featurizers_attributes[str(featurizer_i.__class__.__name__)] = (
                    copy.deepcopy(featurizer_i.__dict__)
                )
        else:
            self.featurizers_attributes = None
        self._featurizer_name = []
        self.smiles = None
        self.x_colnames = None
        self.active_features = None
        self._X_old = None
        self.create()

    @property
    def featurizer_name(self) -> Iterable[Any]:
        return [featurizer.__name__ for featurizer in self.featurizer]

    @featurizer_name.setter
    def featurizer_name(self, value):
        self._featurizer_name = value

    def _validate_column_names(self, cols, col_type):
        """Validate if the columns specified in cols are present in the DataFrame."""
        if len(cols) == 0:
            return

        missing_cols = [col for col in cols if col not in self.df.columns]

        if missing_cols:
            raise ValueError(
                f"The following columns in {col_type} are not present in the DataFrame: {missing_cols}"
            )

    def _validate_column_space(self):
        for ix, col in enumerate(self.x_cols):
            if " " in col:
                new_col = col.replace(" ", "_")
                self.x_cols[ix] = new_col
                self.df.rename(columns={col: new_col}, inplace=True)
                print(
                    f"Warning: Column names cannot have spaces. Column '{col}' has been renamed to '{new_col}'"
                )

    def _validate_column_overlap(self, smiles_cols, x_cols, y_cols):
        smiles_set = set(smiles_cols) if smiles_cols else set()
        x_set = set(x_cols) if x_cols else set()
        y_set = set(y_cols) if y_cols else set()

        overlap_smiles_x = smiles_set & x_set
        overlap_smiles_y = smiles_set & y_set
        overlap_x_y = x_set & y_set

        if len(overlap_smiles_x) > 0:
            raise ValueError(
                f"Overlap found between smiles_cols and x_cols: {overlap_smiles_x}"
            )
        if len(overlap_smiles_y) > 0:
            raise ValueError(
                f"Overlap found between smiles_cols and y_cols: {overlap_smiles_y}"
            )
        if len(overlap_x_y) > 0:
            raise ValueError(f"Overlap found between x_cols and y_cols: {overlap_x_y}")

    def create(self):
        self.df = self.df.reset_index(drop=True)

        if len(self.smiles_cols) == 1:
            # The method featurize_dataframe needs self.smiles to be pd.Series
            self.smiles = self.df[self.smiles_cols[0]]

            # Apply each featurizer to the data
            descriptors_list = [
                featurizer.featurize_dataframe(self.smiles)
                for featurizer in self.featurizer
            ]

            # Concatenate the results from all featurizers
            descriptors = pd.concat(descriptors_list, axis=1)

        elif len(self.smiles_cols) > 1:
            featurized_dfs = []
            for col in self.smiles_cols:
                # Apply each featurizer to the column
                featurized_dfs.append(
                    pd.concat(
                        [
                            featurizer.featurize_dataframe(self.df[[col]])
                            for featurizer in self.featurizer
                        ],
                        axis=1,
                    )
                )

            # Concatenate the results from all SMILES columns and featurizers
            descriptors = pd.concat(featurized_dfs, axis=1)
        else:
            # Case where no smiles were provided
            self.smiles = []
            descriptors = []

        if len(self.x_cols) == 0:
            if len(descriptors) > 0:
                self.X = descriptors
                self.x_colnames = self.X.columns.tolist()
            else:
                raise ValueError(
                    "The design matrix X is empty. Please provide either"
                    "smiles or other descriptors"
                )

        else:
            self.X = pd.concat(
                [self.df[self.x_cols], pd.DataFrame(descriptors)], axis=1
            )
            self.x_colnames = self.X.columns.tolist()

        if not self.y_cols:
            self.y = None
        else:
            self.y = self.df[self.y_cols]

        self.df = pd.concat([self.X, self.y], axis=1)
        self.X.columns = self.X.columns.astype(str)
        self.df.columns = self.df.columns.astype(str)
        self.active_features = self.X.columns.tolist()

    def select_features(self, FeatureSelector=None, SelectionList=None):
        if (FeatureSelector is None and SelectionList is None) or (
            FeatureSelector is not None and SelectionList is not None
        ):
            raise ValueError(
                "Either FeatureSelector or SelectionList must be provided, but not both."
            )

        self._X_old = self.X

        if FeatureSelector is not None:
            # Get all valid feature selection classes from sklearn.feature_selection
            valid_classes = [
                getattr(skfs, name)
                for name in dir(skfs)
                if isinstance(getattr(skfs, name), type)
            ]

            # Check if FeatureSelector is an instance of one of these classes
            if not isinstance(FeatureSelector, tuple(valid_classes)):
                raise ValueError(
                    f"FeatureSelector must be an instance of a valid class from sklearn.feature_selection, but got {type(FeatureSelector)}."
                )
            transformed_X = FeatureSelector.fit_transform(self.X)
            selected_columns_mask = FeatureSelector.get_support()
            self.selected_features = self.X.columns[selected_columns_mask]
            self.X = pd.DataFrame(data=transformed_X, columns=self.selected_features)
        elif SelectionList is not None:
            if not all(item in self.X.columns for item in SelectionList):
                raise ValueError("Provided features not in dataset features")
            else:
                self.X = self.X[SelectionList]
                self.selected_features = SelectionList

    def copy(self):
        """Create a copy of the dataset, including a deep copy of the underlying DataFrame
        and all relevant attributes.
        """
        copied_instance = JaqpotpyDataset(
            df=self.init_df,
            path=self.path,
            y_cols=self.y_cols,
            x_cols=self.x_cols,
            smiles_cols=self.smiles_cols,
            featurizer=self.featurizer,
            task=self.task,
        )
        return copied_instance

    def __get_X__(self):
        return self.X.copy()

    def __get_Y__(self):
        return self.y.copy()

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.df]

    def __getitem__(self, idx):
        selected_x = self.df[self.X].iloc[idx].values
        selected_y = self.df[self.y].iloc[idx].to_numpy()
        return selected_x, selected_y

    def __len__(self):
        return self.df.shape[0]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(smiles={True if self.smiles_cols is not None else False}, "
            f"featurizer={self.featurizer_name})"
        )


if __name__ == "__main__":
    ...
