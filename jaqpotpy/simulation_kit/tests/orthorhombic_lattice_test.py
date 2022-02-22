# """
# Tests for Jaqpotpy Orthorhombic Lattice.
# """
# import unittest
#
# class TestOrthorhombic(unittest.TestCase):
#     """
#     Test CubicLatttice.
#     """
#
#     def setUp(self):
#         """
#         Set up tests.
#         """
#         from jaqpotpy.simulation_kit import SimpleOrthorhombic
#
#         a = 4.2
#         b = 7.9
#         c = 10
#
#         self.cubic = SimpleOrthorhombic(a, b, c)
#         self.lattice = self.cubic.construct()
#
#     def test_atoms(self):
#         assert self.lattice.atoms_per_cell == 1
#
#     def test_params(self):
#         assert self.lattice.a == 4.2
#         assert self.lattice.b == 7.9
#         assert self.lattice.c == 10
#         assert self.lattice.alpha == 90
#         assert self.lattice.beta == 90
#         assert self.lattice.gamma == 90
#
# if __name__ == '__main__':
#     unittest.main()