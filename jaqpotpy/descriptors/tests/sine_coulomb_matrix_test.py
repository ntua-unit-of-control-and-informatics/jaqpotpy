from jaqpotpy.descriptors.material import SineCoulombMatrix
from pymatgen.core import Structure, Lattice
import unittest

class test_SineCoulombMatrix(unittest.TestCase):

      def setUp(self) -> None:
          self.path = './jaqpotpy/test_data/test.extxyz'
          lattice = Lattice.cubic(4.2)
          self.struct = Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
          self.featurizer_flat = SineCoulombMatrix(flatten=True)
          self.featurizer = SineCoulombMatrix()

      def test_featurize_struct(self):
          features_flat = self.featurizer_flat.featurize(self.struct)
          features = self.featurizer.featurize(self.struct)

          assert features_flat.shape == (1, self.featurizer_flat.max_atoms)
          assert features.shape == (1, self.featurizer.max_atoms, self.featurizer.max_atoms, self.featurizer.max_atoms)
          return

      def test_featurize_struct_df(self):
          features_flat = self.featurizer_flat.featurize_dataframe(self.struct)
          features = self.featurizer.featurize_dataframe(self.struct)

          assert all(features == features_flat)
          assert features_flat.shape == (1, self.featurizer_flat.max_atoms)
          return

      def test_featurize_path(self):
          features_flat = self.featurizer_flat.featurize(self.path)
          features = self.featurizer.featurize(self.path)

          assert features_flat.shape == (1, self.featurizer_flat.max_atoms)
          assert features.shape == (1, self.featurizer.max_atoms, self.featurizer.max_atoms, self.featurizer.max_atoms)
          return

      def test_featurize_path_df(self):
          features_flat = self.featurizer_flat.featurize_dataframe(self.path)
          features = self.featurizer.featurize_dataframe(self.path)

          assert all(features == features_flat)
          assert features_flat.shape == (1, self.featurizer_flat.max_atoms)
          return


if __name__ == '__main__':
    unittest.main()