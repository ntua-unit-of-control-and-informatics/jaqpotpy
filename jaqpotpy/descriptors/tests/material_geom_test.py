from jaqpotpy.descriptors.material import GeomDescriptors
import unittest


class test_GeomDesc(unittest.TestCase):

    def setUp(self) -> None:
        
        from jaqpotpy.parsers import PdbParser, MolParser, XyzParser

        # path = 'C:/Users/jason/OneDrive/Documents/PhD - NTUA Process Control/Project 01 - PDB Files/PDB_files/GNP1.pdb'
        #pdb = 'C:/Users/jason/OneDrive/Documents/PhD - NTUA Process Control/Project 01 - PDB Files/'
        #mol = 'C:/Users/jason/Downloads/ChEBI_16716.mol'
        #sdf = 'C:/Users/jason/Downloads/ChEBI_16716.sdf'
        #sdfs = 'C:/Users/jason/Downloads/ChEBI_16732.sdf'
        #folder = 'C:/Users/jason/Downloads/'
        #xyz = 'C:/Users/jason/Downloads/cyclohexane.xyz'
        #extxyz = 'C:/Users/jason/Downloads/lala.extxyz'


        # parser = PdbParser(pdb, 'pdb')
        #parser = MolParser(sdf, 'sdf')
        # parser = XyzParser(extxyz, 'extxyz')
        #
        self.featurizer = GeomDescriptors(parser, ['atomic_number', 'thermal_conductivity'])

    @unittest.skip("Material modelling has not been tested yet in the newest version of jaqpotpy")
    def test(self):
        array = self.featurizer.featurize()
        print(array.shape)
        print(array[0])
        print(self.featurizer._get_column_names())
        return

    @unittest.skip("Material modelling has not been tested yet in the newest version of jaqpotpy")
    def test_df(self):
        df = self.featurizer.featurize_dataframe()
        print('1', df.shape)
        print('2', df.head())
        print('3', df.columns)
        return


if __name__ == '__main__':
    unittest.main()