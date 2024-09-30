"""Topological fingerprints."""

from typing import Dict
import pandas as pd
import numpy as np

from jaqpotpy.utils.types import RDKitMol
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer


class TopologicalFingerprint(MolecularFeaturizer):
    """Circular (Morgan) fingerprints.
    Extended Connectivity Circular Fingerprints compute a bag-of-words style
    representation of a molecule by breaking it into local neighborhoods and
    hashing into a bit vector of the specified size. It is used specifically
    for structure-activity modelling. See [1]_ for more details.

    References:
    ----------
    .. [1] Rogers, David, and Mathew Hahn. "Extended-connectivity fingerprints."
     Journal of chemical information and modeling 50.5 (2010): 742-754.

    Note:
    ----
    This class requires RDKit to be installed.

    Examples:
    --------
    >>> import jaqpotpy as jt
    >>> from rdkit import Chem
    >>> smiles = ['C1=CC=CC=C1']
    >>> # Example 1: (size = 2048, radius = 4)
    >>> featurizer = jt.descriptors.TopologicalFingerprint(size=2048, radius=4)
    >>> features = featurizer.featurize(smiles)


    """

    @property
    def __name__(self):
        return "TopologicalFingerprint"

    def __init__(
        self,
        radius: int = 2,
        size: int = 2048,
        chiral: bool = False,
        bonds: bool = True,
        features: bool = False,
    ):
        """Parameters
        ----------
        radius: int, optional (default 2)
        Fingerprint radius.
        size: int, optional (default 2048)
        Length of generated bit vector.
        chiral: bool, optional (default False)
        Whether to consider chirality in fingerprint generation.
        bonds: bool, optional (default True)
        Whether to consider bond order in fingerprint generation.
        features: bool, optional (default False)
        Whether to use feature information instead of atom information; see
        RDKit docs for more info.
        """
        self.radius = radius
        self.size = size
        self.chiral = chiral
        self.bonds = bonds
        self.features = features
        self.col_names = None

    def _featurize(self, datapoint, convert_nan: bool = True, **kwargs) -> np.ndarray:
        """Calculate circular fingerprint.

        Parameters:datapoint--> rdkit.Chem.rdchem.Mol
        Returns:np.ndarray (A numpy array of circular fingerprint).

        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            datapoint,
            self.radius,
            nBits=self.size,
            useChirality=self.chiral,
            useBondTypes=self.bonds,
            useFeatures=self.features,
        )
        fp = np.asarray(fp, dtype=float)
        return fp

    def __hash__(self):
        return hash(
            (
                self.radius,
                self.size,
                self.chiral,
                self.bonds,
                self.features,
            )
        )

    def __eq__(self, other):
        if not isinstance(self, other.__class__):
            return False
        return (
            self.radius == other.radius
            and self.size == other.size
            and self.chiral == other.chiral
            and self.bonds == other.bonds
            and self.features == other.features
        )

    def featurize_dataframe(
        self, datapoints, convert_nan=False, log_every_n=1000, **kwargs
    ) -> pd.DataFrame:
        features = self.featurize(
            datapoints, convert_nan=True, log_every_n=1000, **kwargs
        )

        self.col_names = [f"Bit_{i}" for i in range(self.size)]
        df = pd.DataFrame(features, columns=self.col_names)
        return df
