# """
# Tests for Jaqpotpy PeriodicTable + Element.
# """
# import unittest
# from jaqpotpy.simulation_kit import PeriodicTable, Element
#
#
# class TestpPeriodicTable(unittest.TestCase):
#     """
#     Test PeriodicTable + Element..
#     """
#
#     def setUp(self):
#         """
#         Set up tests.
#         """
#
#         self.table = PeriodicTable()
#
#
#
#     def test_ptable(self):
#
#         assert self.table.get_tables == ['alembic_version',
#                                          'elements',
#                                          'groups',
#                                          'ionicradii',
#                                          'ionizationenergies',
#                                          'isotopes',
#                                          'oxidationstates',
#                                          'screeningconstants',
#                                          'series']
#         return
#
#     def test_run_query(self):
#         hydrogen = self.table._find_one('SELECT * FROM elements WHERE symbol="H"')
#         assert hydrogen['symbol'] == 'H'
#         assert hydrogen['atomic_number'] == 1
#         return
#
#     def test_element(self):
#         hydrogen = Element('H')
#         params = hydrogen.parameters
#         print(params.evaporation_heat)
#         print(params.fusion_heat)
#         print(params.covalent_radius_bragg, params.covalent_radius_cordero, params.covalent_radius_pyykko, params.covalent_radius_pyykko_double, params.covalent_radius_pyykko_triple)
#         print(type(params.covalent_radius_pyykko), type(params.covalent_radius_pyykko_double), type(params.covalent_radius_pyykko_triple))
#         print('sc',hydrogen.screeningconstants)
#         assert params.symbol == 'H'
#         assert params.atomic_number == 1
#         return
#
#     def test_element_declare(self):
#         hydrogen_1 = Element('H')
#         hydrogen_2 = Element(1)
#
#         assert hydrogen_1.parameters.symbol == hydrogen_2.parameters.symbol
#         assert hydrogen_1.parameters.atomic_number == hydrogen_2.parameters.atomic_number
#         return
#
#
# if __name__ == '__main__':
#     unittest.main()