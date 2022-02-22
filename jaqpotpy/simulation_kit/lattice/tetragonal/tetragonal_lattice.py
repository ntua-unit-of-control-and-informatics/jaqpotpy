# from jaqpotpy.simulation_kit import BravaisLattice
# from typing import Union, Sequence
# import numpy as np
#
# class SimpleTetragonal(BravaisLattice):
#     """
#     Simple Tetragonal Lattice.
#     This class creates a cubic lattice given the lattice parameters 'a' and 'c'
#
#     Examples
#     --------
#     >>> import jaqpotpy as jt
#     >>> sc = jt.simulation_kit.SimpleTetragonal(a = 4.2, b = 9.3, c = 15)
#     >>> cubic_lat = sc.construct()
#     >>> type(cubic_lat)
#     <class 'jaqpotpy.simulation_kit.cubic.cubic.SimpleTetragonal'>
#     """
#
#     def __init__(self, a: float, c: float,array: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray] = []):
#         super().__init__(array, 1.0)
#         self._a = a
#         self._b = a
#         self._c = c
#
#         self._alpha = 90
#         self._beta = 90
#         self._gamma = 90
#
#         self.atoms_inside = 0
#         self.atoms_edge = 0
#         self.atoms_corner = 8
#
#     @property
#     def __name__(self):
#         return 'SimpleTetragonal'
#
#     def __getitem__(self):
#         return self
#
#     def _construct(self) -> BravaisLattice:
#         """
#         Construct tetragonal lattice
#
#         Returns
#         -------
#         jaqpotpy.simulation_kit.SimpleTetragonal
#           A Simple Tetragonal Lattice lattice
#         """
#         lattice = np.array([[self._a, 0, 0], [0, self._a, 0], [0, 0, self._c]])
#         basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#
#         return SimpleTetragonal(self.basis_factor * np.dot(basis, lattice))
#
#     def _atoms_per_cell(self):
#         return self.atoms_inside + self.atoms_edge / 2 + self.atoms_corner / 8
#
#
# class CenteredTetragonal(BravaisLattice):
#     """
#     Centered Tetragonal Lattice.
#     This class creates a cubic lattice given the lattice parameters 'a' and 'c'
#
#     Examples
#     --------
#     >>> import jaqpotpy as jt
#     >>> sc = jt.simulation_kit.CenteredTetragonal(a = 4.2, b = 9.3, c = 15)
#     >>> cubic_lat = sc.construct()
#     >>> type(cubic_lat)
#     <class 'jaqpotpy.simulation_kit.cubic.cubic.CenteredTetragonal'>
#     """
#
#     def __init__(self, a: float, c: float,array: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray] = []):
#         super().__init__(array, 1.0)
#         self._a = a
#         self._b = a
#         self._c = c
#
#         self._alpha = 90
#         self._beta = 90
#         self._gamma = 90
#
#         self.atoms_inside = 1
#         self.atoms_edge = 0
#         self.atoms_corner = 8
#
#     @property
#     def __name__(self):
#         return 'CenteredTetragonal'
#
#     def __getitem__(self):
#         return self
#
#     def _construct(self) -> BravaisLattice:
#         """
#         Construct tetragonal lattice
#
#         Returns
#         -------
#         jaqpotpy.simulation_kit.CenteredTetragonal
#           A Centered Tetragonal Lattice lattice
#         """
#         lattice = np.array([[self._a, 0, 0], [0, self._a, 0], [0, 0, self._c]])
#         basis = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
#
#         return CenteredTetragonal(self.basis_factor * np.dot(basis, lattice))
#
#     def _atoms_per_cell(self):
#         return self.atoms_inside + self.atoms_edge / 2 + self.atoms_corner / 8
