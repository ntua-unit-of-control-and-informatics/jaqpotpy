"""Basic molecular features."""

from typing import Any
import pandas as pd
import numpy as np
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from rdkit.Chem import Descriptors


class RDKitDescriptors(MolecularFeaturizer):
    """RDKit descriptors.

    Examples
    --------
    >>> import jaqpotpy as jt
    >>> smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']
    >>> featurizer = jt.feat.RDKitDescriptors()
    >>> features = featurizer.featurize(smiles)
    """

    @property
    def __name__(self):
        return "RDKitDescriptors"

    def __init__(self, use_fragment=True, ipc_avg=True):
        """Initialize this featurizer.

        Parameters
        ----------
        use_fragment : bool, optional (default=True)
            If True, the return value includes the fragment binary descriptors like 'fr_XXX'.
        ipc_avg : bool, optional (default=True)
            If True, the IPC descriptor calculates with avg=True option.
            Please see this issue: https://github.com/rdkit/rdkit/issues/1527.
        """
        self.use_fragment = use_fragment
        self.ipc_avg = ipc_avg

    def _featurize(self, datapoint, **kwargs) -> np.ndarray:
        """Calculate RDKit descriptors.

        Parameters
        ----------
        datapoint : rdkit.Chem.rdchem.Mol
            The molecule to featurize.

        Returns
        -------
        np.ndarray
            The calculated features as a numpy array.
        """
        descList = self.get_desc_list()
        features = []
        for desc_name, function in descList:
            if desc_name == "Ipc" and self.ipc_avg:
                feature = function(datapoint, avg=True)
            else:
                feature = function(datapoint)
            features.append(feature)
        return np.asarray(features)

    def get_desc_list(self):
        """Get the list of RDKit descriptors.

        Returns
        -------
        list
            A list of tuples containing descriptor names and their corresponding functions.
        """
        descList = []
        try:
            for descriptor, function in Descriptors.descList:
                if self.use_fragment is False and descriptor.startswith("fr_"):
                    continue
                descList.append((descriptor, function))
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")
        return descList

    def get_desc_names(self):
        """Get the names of RDKit descriptors.

        Returns
        -------
        list
            A list of descriptor names.
        """
        descList = self.get_desc_list()
        col_names = [desc_name for desc_name, _ in descList]
        return col_names

    def featurize_dataframe(
        self, datapoints, convert_nan: bool = False, log_every_n=1000, **kwargs
    ) -> pd.DataFrame:
        """Calculate RDKit descriptors for a list of SMILES strings.

        Parameters
        ----------
        datapoints : list of str
            List of SMILES strings to featurize.
        convert_nan : bool, optional (default=False)
            If True, NaN values in the features will be converted.
        log_every_n : int, optional (default=1000)
            Log progress every n datapoints.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the calculated features.
        """
        features = self.featurize(datapoints, convert_nan, log_every_n, **kwargs)
        df = pd.DataFrame(features, columns=self.get_desc_names())
        return df
