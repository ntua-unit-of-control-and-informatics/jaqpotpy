# """
# Tests for Jaqpotpy Composition Lattice.
# """
# import unittest
# from jaqpotpy.simulation_kit import Composition, Element
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
#         self.composition = Composition('Fe2O3')
#         # self.composition = Composition('Fe4O6')
#         # self.composition = Composition({'Fe': 2, 'O': 3})
#         # self.composition = Composition({Element('Fe'): 2, Element('O'): 3})
#
#     def test_dict(self):
#         assert self.composition.as_dict() == {'Fe': 2.0, 'O': 3.0}
#
#     def test_num_atoms(self):
#         assert self.composition.num_atoms == sum(self.composition.as_dict().values())
#
#     def test_reduced_form(self):
#         assert self.composition.reduced_formula == 'Fe2O3'
#
#     def test_add_comp(self):
#         newComp = Composition('FeO')
#         self.composition.__add__(newComp)
#         assert self.composition.reduced_formula == 'Fe3O4'
#
#     def test_avg_electronegativity(self):
#         el = [Element(k).parameters.en_pauling*v for k,v in self.composition.as_dict().items()]
#         avg = sum(el)/sum(self.composition.as_dict().values())
#         assert self.composition.average_electroneg == avg
#
#     def test_total_electrons(self):
#         assert self.composition.total_electrons == sum([Element(k).parameters.atomic_number*v for k,v in self.composition.as_dict().items()])
#
#     def test_fractional_composition(self):
#         fract = self.composition.fractional_composition
#         assert fract.as_dict() == {k: v/self.composition.num_atoms for k, v in self.composition.as_dict().items()}
#
#     def test_reduced_composition(self):
#         reduced = self.composition.reduced_composition
#         # print(reduced.as_dict())
#         assert reduced.as_dict() == {'Fe': 2.0, 'O': 3.0}
#
#     def test_hill_formula(self):
#         assert self.composition.hill_formula == 'Fe2 O3'
#
#     def test_elements(self):
#         print(type(self.composition.elements[0]))
#
#     def test_items(self):
#         print(self.composition._data.items())
#
# if __name__ == '__main__':
#     unittest.main()