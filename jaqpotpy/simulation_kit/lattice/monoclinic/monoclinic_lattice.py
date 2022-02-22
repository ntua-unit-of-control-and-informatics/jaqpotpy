# from jaqpotpy.simulation_kit import BravaisLattice
# from typing import Union, Sequence
# import numpy as np
#
# class SimpleMonoclinic(BravaisLattice):
#     """
#     Simple Monoclinic Lattice.
#     This class creates a cubic lattice given the lattice parameters 'a', 'b', 'c', 'alpha', 'beta' and 'gamma'
#
#     Examples
#     --------
#     >>> import jaqpotpy as jt
#     >>> sc = jt.simulation_kit.SimpleMonoclinic(a = 4.2, b = 9.3, c = 15, aplha=60, beta=60, gamma=60)
#     >>> lattice= sc.construct()
#     >>> type(lattice)
#     <class 'jaqpotpy.simulation_kit.cubic.cubic.SimpleMonoclinic'>
#     """
#
#     def __init__(self, a: float, b: float, c: float, alpha: float,
#                  array: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray] = []):
#         super().__init__(array, 1.0)
#         self._a = a
#         self._b = b
#         self._c = c
#
#         self._alpha = alpha
#         self._beta = 90
#         self._gamma = 90
#
#         self.atoms_inside = 0
#         self.atoms_edge = 0
#         self.atoms_corner = 8
#
#     @property
#     def __name__(self):
#         return 'SimpleMonoclinic'
#
#     def __getitem__(self):
#         return self
#
#     def _construct(self) -> BravaisLattice:
#         """
#         Construct Monoclinic lattice
#
#         Returns
#         -------
#         jaqpotpy.simulation_kit.SimpleMonoclinic
#           A Simple Monoclinic Lattice lattice
#         """
#
#         degree = np.pi / 180.0
#         cosa = np.cos(self._alpha * degree)
#         cosb = np.cos(self._beta * degree)
#         sinb = np.sin(self._beta * degree)
#         cosg = np.cos(self._gamma * degree)
#         sing = np.sin(self._gamma * degree)
#         lattice = np.array([[self._a, 0, 0],
#                             [self._b * cosg, self._b * sing, 0],
#                             [self._c * cosb, self._c * (cosa - cosb * cosg) / sing,
#                              self._c * np.sqrt(sinb ** 2 - ((cosa - cosb * cosg) / sing) ** 2)]])
#         basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#
#         return SimpleMonoclinic(self.basis_factor * np.dot(basis, lattice))
#
#     def _atoms_per_cell(self):
#         return self.atoms_inside + self.atoms_edge / 2 + self.atoms_corner / 8
#
#
#
# class BaseCenteredMonoclinic(BravaisLattice):
#     """
#     Base Centered Monoclinic Lattice.
#     This class creates a cubic lattice given the lattice parameters 'a', 'b', 'c', 'alpha', 'beta' and 'gamma'
#
#     Examples
#     --------
#     >>> import jaqpotpy as jt
#     >>> sc = jt.simulation_kit.BaseCenteredMonoclinic(a = 4.2, b = 9.3, c = 15, aplha=60, beta=60, gamma=60)
#     >>> lattice= sc.construct()
#     >>> type(lattice)
#     <class 'jaqpotpy.simulation_kit.cubic.cubic.BaseCenteredMonoclinic'>
#     """
#
#     def __init__(self, a: float, b: float, c: float, alpha: float,
#                  array: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray] = []):
#         super().__init__(array, 0.5)
#         self._a = a
#         self._b = b
#         self._c = c
#
#         self._alpha = alpha
#         self._beta = 90
#         self._gamma = 90
#
#         self.atoms_inside = 0
#         self.atoms_edge = 0
#         self.atoms_corner = 8
#
#     @property
#     def __name__(self):
#         return 'BaseCenteredMonoclinic'
#
#     def __getitem__(self):
#         return self
#
#     def _construct(self) -> BravaisLattice:
#         """
#         Construct Base Centered Monoclinic lattice
#
#         Returns
#         -------
#         jaqpotpy.simulation_kit.BaseCenteredMonoclinic
#           A Base Centered Monoclinic Lattice lattice
#         """
#
#         degree = np.pi / 180.0
#         cosa = np.cos(self._alpha * degree)
#         cosb = np.cos(self._beta * degree)
#         sinb = np.sin(self._beta * degree)
#         cosg = np.cos(self._gamma * degree)
#         sing = np.sin(self._gamma * degree)
#         lattice = np.array([[self._a, 0, 0],
#                             [self._b * cosg, self._b * sing, 0],
#                             [self._c * cosb, self._c * (cosa - cosb * cosg) / sing,
#                              self._c * np.sqrt(sinb ** 2 - ((cosa - cosb * cosg) / sing) ** 2)]])
#         basis = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]])
#
#         return BaseCenteredMonoclinic(self.basis_factor * np.dot(basis, lattice))
#
#     def _atoms_per_cell(self):
#         return self.atoms_inside + self.atoms_edge / 2 + self.atoms_corner / 8
#
