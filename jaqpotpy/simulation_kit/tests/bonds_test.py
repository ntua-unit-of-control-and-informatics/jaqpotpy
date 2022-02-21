# """
# Tests for Jaqpotpy Bonds.
# """
# import unittest
# from jaqpotpy.simulation_kit import Bond
# from jaqpotpy.simulation_kit.sites.sites import Site, Composition
#
# class TestComposition(unittest.TestCase):
#     """
#     Test Composition.
#     """
#
#     def setUp(self):
#         """
#         Set up tests.
#         """
#         composition = Composition('C')
#         self.site1 = Site(composition, [0, 0, 0])
#
#         composition = Composition('Cl')
#         self.site2 = Site(composition, [0, 0.5, 0])
#
#         self.bond = Bond([self.site1, self.site2])
#
#     def test_bond(self):
#         print(self.bond)
#         # assert self.site.as_dict() == {'name': 'Fe', 'species': [{'occu': 1, 'element': 'Fe'}], 'xyz': [0.0, 0.0, 0.0], 'properties': {}, '@module': 'jaqpotpy.simulation_kit.utils.sites', '@class': 'Site'}
#         return
#
#     def test_length(self):
#         print(self.bond.bond_length())
#     #
#     # def test_items(self):
#     #     print(self.site.species.keys())
#
# if __name__ == '__main__':
#     unittest.main()