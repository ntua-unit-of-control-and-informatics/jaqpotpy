"""
Tests for Jaqpotpy Models.
"""
import unittest
from jaqpotpy.parsers.pdb_parser import PdbParser


class TestParsers(unittest.TestCase):
    """
    Test PdbParser.
    """

    def setUp(self):
        """
        Set up tests.
        """

        #pdb = 'C:/Users/jason/OneDrive/Documents/PhD - NTUA Process Control/Project 01 - PDB Files/PDB_files/GNP1.pdb'
        #path = 'C:/Users/jason/OneDrive/Documents/PhD - NTUA Process Control/Project 01 - PDB Files/PDB_files/'

        #self.parser = PdbParser(pdb, 'pdb')

    @unittest.skip("This test needs refactoring")  
    def test_file(self):
        """
        Test pdb file.
        """
        # test = self.parser.parse()
        # pdb = next(test)
        # print(pdb.atoms.elements)
        # assert pdb.atoms.elements
        return

    @unittest.skip("This test needs refactoring") 
    def test_path(self):
        """
        Test pdb folder.
        """
        # test = self.parser.parse()
        # stop = ''
        #
        # while stop.lower() != 'q':
        #     pdb = next(test)
        #     print(pdb.atoms.elements)
        #     print(self.parser.files_)
        #     stop = input('q to stop')

        return

    @unittest.skip("This test needs refactoring") 
    def test_df_file(self):
        #df = self.parser.parse_dataframe()
        # pdb = next(test)
        #print(df)
        return

    @unittest.skip("This test needs refactoring") 
    def test_xyz(self):
        #print(self.parser.to_xyz())
        return

if __name__ == '__main__':
    unittest.main()