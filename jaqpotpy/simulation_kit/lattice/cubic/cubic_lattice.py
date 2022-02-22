# from jaqpotpy.simulation_kit import BravaisLattice
# from typing import Union, Sequence
# import numpy as np
#
# class SimpleCubic(BravaisLattice):
#     """
#     Simple Cubic Lattice.
#     This class creates a cubic lattice given the lattice parameter 'a'
#
#
#     Examples
#     --------
#     >>> import jaqpotpy as jt
#     >>> sc = jt.simulation_kit.CubicLattice(a = 4.2)
#     >>> cubic_lat = sc.construct()
#     >>> type(cubic_lat)
#     <class 'jaqpotpy.simulation_kit.cubic.cubic.SimpleCubic'>
#     """
#
#     def __init__(self, a: float, array: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray] = []):
#         super().__init__(array, 1.0)
#         self._a = a
#         self._b = a
#         self._c = a
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
#         return 'SimpleCubic'
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
#         jaqpotpy.simulation_kit.SimpleCubic
#           A Cubic lattice
#         """
#         return SimpleCubic(self._a, [
#             [self._a * self.basis_factor, 0.0, 0.0],
#             [0.0, self._a * self.basis_factor, 0.0],
#             [0.0, 0.0, self._a * self.basis_factor]
#         ])
#
#     def _atoms_per_cell(self):
#         return self.atoms_inside + self.atoms_edge / 2 + self.atoms_corner / 8
#
#
# class FaceCenteredCubic(BravaisLattice):
#     """
#     Face Centered Cubic (FCC) Lattice.
#     This class creates a cubic lattice given the lattice parameter 'a'
#
#     Examples
#     --------
#     >>> import jaqpotpy as jt
#     >>> fcc = jt.simulation_kit.FaceCenteredCubic(a = 4.2)
#     >>> cubic_lat = fcc.construct()
#     >>> type(cubic_lat)
#     <class 'jaqpotpy.simulation_kit.cubic.cubic.FaceCenteredCubic'>
#     """
#
#     def __init__(self, a: float, array: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray] = []):
#         super().__init__(array, 0.5)
#         self._a = a
#         self._b = a
#         self._c = a
#
#         self._alpha = 90
#         self._beta = 90
#         self._gamma = 90
#
#         self.atoms_inside = 0
#         self.atoms_edge = 6
#         self.atoms_corner = 8
#
#     @property
#     def __name__(self):
#         return 'FaceCenteredCubic'
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
#         jaqpotpy.simulation_kit.FaceCenteredCubic
#           A Face Centered Cubic lattice
#         """
#         return FaceCenteredCubic(self._a, [
#             [0.0, self._a * self.basis_factor, self._a * self.basis_factor],
#             [self._a * self.basis_factor, 0.0, self._a * self.basis_factor],
#             [self._a * self.basis_factor, self._a * self.basis_factor, 0.0]
#         ])
#
#     def _atoms_per_cell(self):
#         return self.atoms_inside + self.atoms_edge / 2 + self.atoms_corner / 8
#
#
# class BodyCenteredCubic(BravaisLattice):
#     """
#     Body Centered Cubic (BCC) Lattice.
#     This class creates a cubic lattice given the lattice parameter 'a'
#
#     Examples
#     --------
#     >>> import jaqpotpy as jt
#     >>> fcc = jt.simulation_kit.FaceCenteredCubic(a = 4.2)
#     >>> cubic_lat = fcc.construct()
#     >>> type(cubic_lat)
#     <class 'jaqpotpy.simulation_kit.cubic.cubic.FaceCenteredCubic'>
#     """
#
#     def __init__(self, a: float, array: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray] = []):
#         super().__init__(array, 0.5)
#         self._a = a
#         self._b = a
#         self._c = a
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
#         return 'BodyCenteredCubic'
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
#         jaqpotpy.simulation_kit.FaceCenteredCubic
#           A Body Centered Cubic lattice
#         """
#         return BodyCenteredCubic(self._a, [
#             [-self._a * self.basis_factor, self._a * self.basis_factor, self._a * self.basis_factor],
#             [self._a * self.basis_factor, -self._a * self.basis_factor, self._a * self.basis_factor],
#             [self._a * self.basis_factor, self._a * self.basis_factor, -self._a * self.basis_factor]
#         ])
#
#     def _atoms_per_cell(self):
#         return self.atoms_inside + self.atoms_edge / 2 + self.atoms_corner / 8
