import unittest
from pymatgen.core import Structure, Lattice
from jaqpotpy.descriptors.graph import GraphData
from jaqpotpy.descriptors.material import CrystalGraphCNN



class test_CrystalGraphCNN(unittest.TestCase):

    def setUp(self) -> None:
        self.path = 'C:/Users/jason/OneDrive/Documents/GitHub/jaqpotpy/jaqpotpy/test_data/test.extxyz'
        self.featurizer = CrystalGraphCNN()
        lattice = Lattice.cubic(4.2)
        self.struct = Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


      # def test_path(self):
      #
      #   features = self.featurizer.featurize(self.path)
      #   assert isinstance(features[0], GraphData)
      #   return

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_struct(self):
        features = self.featurizer.featurize(self.struct)
        assert isinstance(features[0], GraphData)
        return


if __name__ == '__main__':
    unittest.main()
