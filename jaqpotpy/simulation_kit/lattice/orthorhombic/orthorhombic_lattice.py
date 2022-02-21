# from jaqpotpy.simulation_kit import BravaisLattice
# from typing import Union, Sequence
# import numpy as np
#
# class SimpleOrthorhombic(BravaisLattice):
#     """
#     Simple Orthorhombic Lattice.
#     This class creates a cubic lattice given the lattice parameters 'a', 'b' and 'c'
#
#     Examples
#     --------
#     >>> import jaqpotpy as jt
#     >>> sc = jt.simulation_kit.SimpleOrthorhombic(a = 4.2, b = 9.3, c = 15)
#     >>> lattice = sc.construct()
#     >>> type(lattice)
#     <class 'jaqpotpy.simulation_kit.cubic.cubic.SimpleOrthorhombic'>
#     """
#
#     def __init__(self, a: float, b: float, c: float,array: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray] = []):
#         super().__init__(array, 1.0)
#         self._a = a
#         self._b = b
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
#         return 'SimpleOrthorhombic'
#
#     def __getitem__(self):
#         return self
#
#     def _construct(self) -> BravaisLattice:
#         """
#         Construct cubic lattice
#
#         Returns
#         -------
#         jaqpotpy.simulation_kit.SimpleOrthorhombic
#           A Simple Orthorhombic lattice
#         """
#         lattice = np.array([[self._a, 0, 0], [0, self._b, 0], [0, 0, self._c]])
#         basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#
#         return SimpleOrthorhombic(self.basis_factor * np.dot(basis, lattice))
#
#     def _atoms_per_cell(self):
#         return self.atoms_inside + self.atoms_edge / 2 + self.atoms_corner / 8
#
# class BaseCenteredOrthorhombic(BravaisLattice):
#     """
#     Base Centered Orthorhombic Lattice.
#     This class creates a cubic lattice given the lattice parameters 'a', 'b' and 'c'
#
#     Examples
#     --------
#     >>> import jaqpotpy as jt
#     >>> sc = jt.simulation_kit.BaseCenteredOrthorhombic(a = 4.2, b = 9.3, c = 15)
#     >>> lattice = sc.construct()
#     >>> type(lattice)
#     <class 'jaqpotpy.simulation_kit.cubic.cubic.BaseCenteredOrthorhombic'>
#     """
#
#     def __init__(self, a: float, b: float, c: float,array: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray] = []):
#         super().__init__(array, 0.5)
#         self._a = a
#         self._b = b
#         self._c = c
#
#         self._alpha = 90
#         self._beta = 90
#         self._gamma = 90
#
#         self.atoms_inside = 0
#         self.atoms_edge = 2
#         self.atoms_corner = 8
#
#     @property
#     def __name__(self):
#         return 'BaseCenteredOrthorhombic'
#
#     def __getitem__(self):
#         return self
#
#     def _construct(self) -> BravaisLattice:
#         """
#         Construct cubic lattice
#
#         Returns
#         -------
#         jaqpotpy.simulation_kit.BaseCenteredOrthorhombic
#           A Base Centered Orthorhombic lattice
#         """
#         lattice = np.array([[self._a, 0, 0], [0, self._b, 0], [0, 0, self._c]])
#         basis = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]])
#
#         return BaseCenteredOrthorhombic(self.basis_factor * np.dot(basis, lattice))
#
#     def _atoms_per_cell(self):
#         return self.atoms_inside + self.atoms_edge / 2 + self.atoms_corner / 8
#
#
# class BodyCenteredOrthorhombic(BravaisLattice):
#     """
#     Body Centered Orthorhombic Lattice.
#     This class creates a cubic lattice given the lattice parameters 'a', 'b' and 'c'
#
#     Examples
#     --------
#     >>> import jaqpotpy as jt
#     >>> sc = jt.simulation_kit.BodyCenteredOrthorhombic(a = 4.2, b = 9.3, c = 15)
#     >>> lattice = sc.construct()
#     >>> type(lattice)
#     <class 'jaqpotpy.simulation_kit.cubic.cubic.BodyCenteredOrthorhombic'>
#     """
#
#     def __init__(self, a: float, b: float, c: float,array: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray] = []):
#         super().__init__(array, 0.5)
#         self._a = a
#         self._b = b
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
#         return 'BodyCenteredOrthorhombic'
#
#     def __getitem__(self):
#         return self
#
#     def _construct(self) -> BravaisLattice:
#         """
#         Construct cubic lattice
#
#         Returns
#         -------
#         jaqpotpy.simulation_kit.BodyCenteredOrthorhombic
#           A Body Centered Orthorhombic lattice
#         """
#         lattice = np.array([[self._a, 0, 0], [0, self._b, 0], [0, 0, self._c]])
#         basis = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
#
#         return BodyCenteredOrthorhombic(self.basis_factor * np.dot(basis, lattice))
#
#     def _atoms_per_cell(self):
#         return self.atoms_inside + self.atoms_edge / 2 + self.atoms_corner / 8
#
#
# class FaceCenteredOrthorhombic(BravaisLattice):
#     """
#     Face Centered Orthorhombic Lattice.
#     This class creates a cubic lattice given the lattice parameters 'a', 'b' and 'c'
#
#     Examples
#     --------
#     >>> import jaqpotpy as jt
#     >>> sc = jt.simulation_kit.FaceCenteredOrthorhombic(a = 4.2, b = 9.3, c = 15)
#     >>> lattice = sc.construct()
#     >>> type(lattice)
#     <class 'jaqpotpy.simulation_kit.cubic.cubic.FaceCenteredOrthorhombic'>
#     """
#
#     def __init__(self, a: float, b: float, c: float,array: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray] = []):
#         super().__init__(array, 0.5)
#         self._a = a
#         self._b = b
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
#         return 'FaceCenteredOrthorhombic'
#
#     def __getitem__(self):
#         return self
#
#     def _construct(self) -> BravaisLattice:
#         """
#         Construct cubic lattice
#
#         Returns
#         -------
#         jaqpotpy.simulation_kit.FaceCenteredOrthorhombic
#           A Face Centered Orthorhombic lattice
#         """
#         lattice = np.array([[self._a, 0, 0], [0, self._b, 0], [0, 0, self._c]])
#         basis = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
#
#         return FaceCenteredOrthorhombic(self.basis_factor * np.dot(basis, lattice))
#
#     def _atoms_per_cell(self):
#         return self.atoms_inside + self.atoms_edge / 2 + self.atoms_corner / 8
#
