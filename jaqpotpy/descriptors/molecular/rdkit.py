"""Basic molecular features."""

from typing import Any
import pandas as pd
import numpy as np
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from rdkit.Chem import Descriptors


class RDKitDescriptors(MolecularFeaturizer):
    """RDKit descriptors.
    Examples:
    --------
    >>> import jaqpotpy as jt
    >>> smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']
    >>> featurizer = dc.feat.RDKitDescriptors()
    >>> features = featurizer.featurize(smiles)
    """

    @property
    def __name__(self):
        return "RDKitDescriptors"

    def __init__(self, use_fragment=True, ipc_avg=True):
        """Initialize this featurizer.

        Parameters
        ----------
        use_fragment: bool, optional (default True)
          If True, the return value includes the fragment binary descriptors like 'fr_XXX'.
        ipc_avg: bool, optional (default True)
          If True, the IPC descriptor calculates with avg=True option.
          Please see this issue: https://github.com/rdkit/rdkit/issues/1527.

        """
        self.use_fragment = use_fragment
        self.ipc_avg = ipc_avg

    def _featurize(self, datapoint, **kwargs) -> np.ndarray:
        """Calculate RDKit descriptors.

        Parameters:datapoint --> rdkit.Chem.rdchem.Mol
        Returns: features --> np.ndarray
        -------
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
        descList = self.get_desc_list()
        col_names = [desc_name for desc_name, _ in descList]
        return col_names

    def featurize_dataframe(
        self, datapoints, convert_nan: bool = False, log_every_n=1000, **kwargs
    ) -> pd.DataFrame:
        """Calculate Mordred descriptors.
        Parameters: datapoints --> list of SMILES
        Returns: df --> pd.DataFrame
            - If ignore_3D is True, the length is 1613.
            - If ignore_3D is False, the length is 1826.
        """
        features = self.featurize(datapoints, convert_nan, log_every_n, **kwargs)
        df = pd.DataFrame(features, columns=self.get_desc_names())
        return df
