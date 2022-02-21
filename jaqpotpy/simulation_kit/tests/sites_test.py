# """
# Tests for Jaqpotpy Sites.
# """
# import unittest
# from jaqpotpy.simulation_kit import Composition, Site
#
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
#         self.composition = Composition('Fe')
#         self.site = Site(self.composition, [0,0,0])
#         # self.composition = Composition({'Fe': 2, 'O': 3})
#         # self.composition = Composition({Element('Fe'): 2, Element('O'): 3})
#
#     def test_as_dict(self):
#         # print(self.site.as_dict())
#         assert self.site.as_dict() == {'name': 'Fe', 'species': [{'occu': 1, 'element': 'Fe'}], 'xyz': [0.0, 0.0, 0.0], 'properties': {}, '@module': 'jaqpotpy.simulation_kit.utils.sites', '@class': 'Site'}
#
#         return
#
#     def test_from_dict(self):
#         # print(self.site.as_dict())
#         site2 = Site.from_dict({
#                         'name': 'FeO',
#                         'species': [{'element': 'Fe', 'occu': 0.5},{'element': 'O', 'occu': 0.5}],
#                         'xyz': [0.0, 0.0, 0.0],
#                         'properties': {}
#                     })
#
#         assert site2.as_dict() == {'name': 'Fe:0.000, O:0.000', 'species': [{'occu': 0.5, 'element': 'Fe'}, {'occu': 0.5, 'element': 'O'}], 'xyz': [0.0, 0.0, 0.0], 'properties': {}, '@module': 'jaqpotpy.simulation_kit.utils.sites', '@class': 'Site'}
#
#         return
#
#     def test_items(self):
#         print(self.site.species.keys())
#
# if __name__ == '__main__':
#     unittest.main()