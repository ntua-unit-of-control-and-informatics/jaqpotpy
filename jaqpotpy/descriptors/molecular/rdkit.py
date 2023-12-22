"""
Basic molecular features.
"""
from typing import Any
import numpy as np
from jaqpotpy.descriptors.base_classes import MolecularFeaturizer
from rdkit.Chem import Descriptors
import pickle
# import dill as pickle

RDKitMol = Any


class Wrapper(object):
    def __init__(self, method_name, module_name):
        self.method_name = method_name
        self.module_name = module_name

    def __call__(self, *args, **kwargs):
        method = __import__(self.module_name, globals(), locals(), [self.method_name,])
        return method(*args, **kwargs)


class RDKitDescriptors(MolecularFeaturizer):
    """RDKit descriptors.
      This class computes a list of chemical descriptors like
      molecular weight, number of valence electrons, maximum and
      minimum partial charge, etc using RDKit.
      Attributes
      ----------
      descriptors: List[str]
        List of RDKit descriptor names used in this class.
      Note
      ----
      This class requires RDKit to be installed.
      Examples
      --------
      >>> import jaqpotpy as jt
      >>> smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']
      >>> featurizer = dc.feat.RDKitDescriptors()
      >>> features = featurizer.featurize(smiles)
      >>> type(features[0])
      <class 'numpy.ndarray'>
      >>> features[0].shape
      (208,)
      """

    def pick(self):
        with open("test.picl", "wb") as p:
            pickle.dump(self, p)
            # pickle.dump(self, p)

    @classmethod
    def unpick(cls, obj):
        with open(obj, "rb") as p:
            return pickle.load(p)

    @property
    def __name__(self):
        return 'RDKitDescriptors'

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
        self.descriptors = []
        self.descList = []

    def __getitem__(self):
        return self

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
        """
        Calculate RDKit descriptors.
        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          1D array of RDKit descriptors for `mol`.
          The length is `len(self.descriptors)`.
        """
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )
        # initialize
        if len(self.descList) == 0:
            try:
                for descriptor, function in Descriptors.descList:
                    if self.use_fragment is False and descriptor.startswith('fr_'):
                        continue
                    self.descriptors.append(descriptor)
                    self.descList.append((descriptor, function))
            except ModuleNotFoundError:
                raise ImportError("This class requires RDKit to be installed.")

        # check initialization
        assert len(self.descriptors) == len(self.descList)

        features = []
        for desc_name, function in self.descList:
            if desc_name == 'Ipc' and self.ipc_avg:
                feature = function(datapoint, avg=True)
            else:
                feature = function(datapoint)
            features.append(feature)
        return np.asarray(features)

    def _get_column_names(self, **kwargs) -> list:
        descriptors = []
        try:
            for descriptor, function in Descriptors.descList:
                if self.use_fragment is False and descriptor.startswith('fr_'):
                    continue
                descriptors.append(descriptor)
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")
        return descriptors

    def _featurize_dataframe(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
        """
        Calculate RDKit descriptors.
        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          1D array of RDKit descriptors for `mol`.
          The length is `len(self.descriptors)`.
        """
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )
        # initialize
        if len(self.descList) == 0:
            try:
                for descriptor, function in Descriptors.descList:
                    if self.use_fragment is False and descriptor.startswith('fr_'):
                        continue
                    self.descriptors.append(descriptor)
                    self.descList.append((descriptor, function))
            except ModuleNotFoundError:
                raise ImportError("This class requires RDKit to be installed.")

        # check initialization
        assert len(self.descriptors) == len(self.descList)

        features = []
        for desc_name, function in self.descList:
            if desc_name == 'Ipc' and self.ipc_avg:
                feature = function(datapoint, avg=True)
            else:
                feature = function(datapoint)
            features.append(feature)
        return np.asarray(features)
