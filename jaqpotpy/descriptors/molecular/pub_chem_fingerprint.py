import numpy as np

from jaqpotpy.utils.types import RDKitMol
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer


class PubChemFingerprint(MolecularFeaturizer):
    """PubChem Fingerprint.
  The PubChem fingerprint is a 881 bit structural key,
  which is used by PubChem for similarity searching.
  Please confirm the details in [1]_.
  References
  ----------
  .. [1] ftp://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.pdf
  Note
  -----
  This class requires RDKit and PubChemPy to be installed.
  PubChemPy use REST API to get the fingerprint, so you need the internet access.
  Examples
  --------
  >>> import jaqpotpy as jt
  >>> smiles = ['CCC']
  >>> featurizer = jt.descriptors.PubChemFingerprint()
  >>> features = featurizer.featurize(smiles)
  >>> type(features[0])
  <class 'numpy.ndarray'>
  >>> features[0].shape
  (881,)
  """

    @property
    def __name__(self):
        return 'PubChemFingerprint'

    def __init__(self):
        """Initialize this featurizer."""
        try:
            from rdkit import Chem  # noqa
            import pubchempy as pcp  # noqa
        except ModuleNotFoundError:
            raise ImportError("This class requires PubChemPy to be installed.")

        self.get_pubchem_compounds = pcp.get_compounds

    def __getitem__(self):
        return self

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
        """
    Calculate PubChem fingerprint.
    Parameters
    ----------
    datapoint: rdkit.Chem.rdchem.Mol
      RDKit Mol object
    Returns
    -------
    np.ndarray
      1D array of RDKit descriptors for `mol`. The length is 881.
    """
        try:
            from rdkit import Chem
            import pubchempy as pcp
        except ModuleNotFoundError:
            raise ImportError("This class requires PubChemPy to be installed.")
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        smiles = Chem.MolToSmiles(datapoint)
        pubchem_compound = pcp.get_compounds(smiles, 'smiles')[0]
        feature = [int(bit) for bit in pubchem_compound.cactvs_fingerprint]
        return np.asarray(feature)
    
    def _get_column_names(self, **kwargs) -> list:
        descriptors_length = 881
        descriptors = []
        for i in range(descriptors_length):
            descriptors.append("f" + str(i))
        return descriptors

    def _featurize_dataframe(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
        return self._featurize(datapoint, **kwargs)

