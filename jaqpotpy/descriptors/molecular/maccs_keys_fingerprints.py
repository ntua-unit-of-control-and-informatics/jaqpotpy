import numpy as np
import pandas as pd
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer


class MACCSKeysFingerprint(MolecularFeaturizer):
    """MACCS Keys Fingerprint.
    The MACCS (Molecular ACCess System) keys are one of the most commonly used structural keys.

    Examples:
    --------
    >>> import jaqpotpy as jp
    >>> smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    >>> featurizer = jp.descriptors.molecular.MACCSKeysFingerprint()
    >>> features = featurizer.featurize([smiles])
    >>> type(features[0])
    <class 'numpy.ndarray'>
    >>> features[0].shape
    (167,)

    References:
    ----------
    .. [1] Durant, Joseph L., et al. "Reoptimization of MDL keys for use in drug discovery."
       Journal of chemical information and computer sciences 42.6 (2002): 1273-1280.
    .. [2] https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/MACCSkeys.py

    Note:
    ----
    This class requires RDKit to be installed.

    """

    @property
    def __name__(self):
        return "MACCSKeysFingerprint"

    def __init__(self):
        """Initialize this featurizer."""
        self.col_names = None

    def _featurize(self, datapoint, **kwargs) -> np.ndarray:
        """Calculate MACCS keys fingerprint.

        Parameters
        ----------
        datapoint: SMILES

        Returns
        -------
        np.ndarray
          1D array of RDKit descriptors for `mol`. The length is 167.

        """

        try:
            calculator = GetMACCSKeysFingerprint
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        fp = calculator(datapoint)
        array = np.zeros((0,), dtype=np.int8)
        ConvertToNumpyArray(fp, array)
        return np.asarray(array)

    def featurize_dataframe(
        self, datapoints, convert_nan=False, log_every_n=1000, **kwargs
    ) -> pd.DataFrame:
        features = self.featurize(
            datapoints, convert_nan=True, log_every_n=1000, **kwargs
        )
        self.col_names = [f"f{i}" for i in range(167)]
        df = pd.DataFrame(features, columns=self.col_names)
        return df
