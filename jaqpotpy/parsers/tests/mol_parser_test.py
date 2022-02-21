"""
Tests for Jaqpotpy Models.
"""
import unittest



class TestParsers(unittest.TestCase):
    """
    Test MolParser.
    """

    def setUp(self):
        """
        Set up tests.
        """
        from jaqpotpy.parsers.mol_parser.mol_parser import MolParser

        mol = 'C:/Users/jason/Downloads/ChEBI_16716.mol'
        sdf = 'C:/Users/jason/Downloads/ChEBI_16716.sdf'
        sdfs = 'C:/Users/jason/Downloads/ChEBI_16732.sdf'
        folder = 'C:/Users/jason/Downloads/'

        self.parser = MolParser(sdfs, 'sdf')

    def test_mol(self):
        """
        Test mol file.
        """
        # test = self.parser.parse()
        # mol = next(test)
        # print(mol)
        return

    def test_sdf(self):
        """
        Test sdf file.
        """
        # test = self.parser.parse()
        # sdf = next(test)
        # print(sdf)
        # # assert pdb.atoms.elements
        return

    def test_sdfs(self):
        """
        Test sdf file.
        """
        # test = self.parser.parse()
        # sdf1 = next(test)
        # print(sdf1)
        # input()
        # sdf2 = next(test)
        # print(sdf2)

        # assert pdb.atoms.elements
        return

    def test_path(self):
        """
        Test pdb folder.
        """
        # test = self.parser.parse()
        # stop = ''
        #
        # while stop.lower() != 'q':
        #     mol = next(test)
        #     print(mol)
        #     print(self.parser.files_[-1])
        #     stop = input('q to stop')
        return

    def test_df_file(self):
        df = self.parser.parse_dataframe()
        print(df[0],'\n\n', df[1])
        return


if __name__ == '__main__':
    unittest.main()