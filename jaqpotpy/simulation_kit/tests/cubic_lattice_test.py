# """
# Tests for Jaqpotpy Cubic Lattice.
# """
# import unittest
#
# class TestCubic(unittest.TestCase):
#     """
#     Test CubicLatttice.
#     """
#
#     def setUp(self):
#         """
#         Set up tests.
#         """
#         from jaqpotpy.simulation_kit import BodyCenteredCubic
#
#         a = 4.2
#
#         self.cubic = BodyCenteredCubic(a)
#         self.lattice = self.cubic.construct()
#
#     def test_atoms(self):
#         assert self.lattice.atoms_per_cell == 4
#
#     def test_params(self):
#         assert self.lattice.basis_factor == 0.5
#         assert self.lattice.a == 4.2
#         assert self.lattice.b == 4.2
#         assert self.lattice.c == 4.2
#         assert self.lattice.alpha == 90
#         assert self.lattice.beta == 90
#         assert self.lattice.gamma == 90
#
#
#
# if __name__ == '__main__':
#     unittest.main()