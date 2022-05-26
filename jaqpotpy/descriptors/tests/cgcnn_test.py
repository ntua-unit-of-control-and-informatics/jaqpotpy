from jaqpotpy.descriptors.graph import GraphData
from jaqpotpy.descriptors.material import CrystalGraphCNN
import unittest
from pymatgen.core import Structure, Lattice


class test_CrystalGraphCNN(unittest.TestCase):

      def setUp(self) -> None:
          self.path = './jaqpotpy/test_data/test.extxyz'
          self.featurizer = CrystalGraphCNN()
          lattice = Lattice.cubic(4.2)
          self.struct = Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


      def test_path(self):

        features = self.featurizer.featurize(self.path)
        assert isinstance(features[0], GraphData)
        return


      def test_struct(self):

        features = self.featurizer.featurize(self.struct)
        assert isinstance(features[0], GraphData)
        return


if __name__ == '__main__':
    unittest.main()