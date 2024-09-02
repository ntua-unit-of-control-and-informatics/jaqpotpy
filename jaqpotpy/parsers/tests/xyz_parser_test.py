"""
Tests for Jaqpotpy Models.
"""

import unittest


class TestParsers(unittest.TestCase):
    """
    Test XyzParser.
    """

    def setUp(self):
        """
        Set up tests.
        """
        from jaqpotpy.parsers.xyz_parser import XyzParser

        # xyz = 'C:/Users/jason/Downloads/cyclohexane.xyz'
        # extxyz = 'C:/Users/jason/Downloads/lala.extxyz'
        # path = 'C:/Users/jason/Downloads'
        # self.parser = XyzParser(xyz, ['xyz', 'extxyz'])

    @unittest.skip("This test needs refactoring")
    def test_xyz(self):
        """
        Test xyz file.
        """
        # test = self.parser.parse()
        # xyz = next(test)
        # print(xyz)
        return
        # assert pdb.atoms.elements

    @unittest.skip("This test needs refactoring")
    def test_extxyz(self):
        """
        Test extxyz file.
        """
        # test = self.parser.parse()
        # extxyz = next(test)
        # print(extxyz)
        # assert pdb.atoms.elements
        return

    @unittest.skip("This test needs refactoring")
    def test_path(self):
        """
        Test xyz folder.
        """
        # test = self.parser.parse()
        # stop = ''
        #
        # while stop.lower() != 'q':
        #     xyz = next(test)
        #     print(xyz)
        #     print(self.parser.files_[-1])
        #     stop = input('q to stop')
        return

    @unittest.skip("This test needs refactoring")
    def test_df_file(self):
        # df = self.parser.parse_dataframe()
        # print(df)
        return


if __name__ == "__main__":
    unittest.main()
