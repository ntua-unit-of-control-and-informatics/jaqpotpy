from typing import Callable, List, Optional, Any
import numpy as np
import pandas as pd
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
# warnings.filterwarnings("ignore", category=DeprecationWarning)


class MordredDescriptors(MolecularFeaturizer):
    """Mordred descriptors.
       descriptors: List[str]
       List of Mordred descriptor names used in this class.

    References:
    ----------
    .. [1] Moriwaki, Hirotomo, et al. "Mordred: a molecular descriptor calculator."
       Journal of cheminformatics 10.1 (2018): 4.
    .. [2] http://mordred-descriptor.github.io/documentation/master/descriptors.html


    Examples:
    --------
    >>> import jaqpotpy as jt
    >>> smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']
    >>> featurizer = jt.descriptors.MordredDescriptors()
    >>> features = featurizer.featurize(smiles)
    """

    @property
    def __name__(self):
        return "MordredDescriptors"

    def __init__(self, ignore_3D: bool = True):
        """Parameters
        ----------
        ignore_3D: bool, optional (default True)
          Whether to use 3D information or not.

        """
        self.col_names = None
        self.ignore_3D = ignore_3D
        self.calc: Optional[Callable] = None
        self.descriptors: Optional[List] = None

    def _featurize(self, datapoint, convert_nan: bool = True, **kwargs) -> np.ndarray:
        """Calculate Mordred descriptors.
        Parameters:datapoint: rdkit.Chem.rdchem.Mol
        Returns: np.ndarray
          1D array of Mordred descriptors for `mol`.
          If ignore_3D is True, the length is 1613.
          If ignore_3D is False, the length is 1826.

        """
        if self.calc is None:
            try:
                from mordred import Calculator, descriptors, is_missing

                self.is_missing = is_missing
                self.calc = Calculator(descriptors, ignore_3D=self.ignore_3D)
                self.descriptors = list(descriptors.__all__)
            except ModuleNotFoundError:
                raise ImportError("This class requires Mordred to be installed.")

        feature = self.calc(datapoint)
        self.col_names = [key for key in feature.keys()]
        # convert errors to zero
        if convert_nan:
            feature = [
                -1000.0 if self.is_missing(val) or isinstance(val, str) else val
                for val in feature
            ]
        return np.asarray(feature)

    def featurize_dataframe(
        self, datapoints, convert_nan: bool = True, log_every_n=1000, **kwargs
    ) -> pd.DataFrame:
        """Calculate Mordred descriptors.
        Parameters: datapoints --> list of SMILES
        Returns: df --> pd.DataFrame
            - If ignore_3D is True, the length is 1613.
            - If ignore_3D is False, the length is 1826.
        """
        features = self.featurize(datapoints, convert_nan, log_every_n, **kwargs)
        df = pd.DataFrame(features, columns=self.col_names)
        return df
