# import inspect
# import logging
# import numpy as np
# import pandas as pd
# from typing import Any, Iterable, Union, List, Generator, Sequence, Optional, Tuple, Dict
# import os
#
# logger = logging.getLogger(__name__)
# _print_threshold = 10
#
# class BravaisLattice(object):
#   """
#   Abstract class for a lattice.
#   This class is abstract and cannot be invoked directly. You'll
#   likely only interact with this class if you're a developer. In
#   that case, you might want to make a child class which
#   implements the `_construct` method for constructing a lattice.
#
#   Note
#   ----
#   In general, it is assumed that length units are in Angstroms and
#   angles are in degrees unless otherwise stated.
#   """
#
#   def __init__(self, array: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray], basis_factor: float):
#       """
#       Create a lattice from a sequence of 9 numbers.
#
#       Parameters
#       ----------
#       array: Sequence of numbers in any form. Examples of acceptable
#           --> An actual numpy array.
#           --> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
#           --> [1, 0, 0 , 0, 1, 0, 0, 0, 1]
#           --> (1, 0, 0, 0, 1, 0, 0, 0, 1)
#           Each row should correspond to a lattice vector.
#           E.g., [[10, 0, 0], [20, 10, 0], [0, 0, 30]] specifies a lattice
#           with lattice vectors [10, 0, 0], [20, 10, 0] and [0, 0, 30].
#
#       Note
#       ----
#       The sequence is assumed to be read one row at a time.
#       Each row represents one lattice vector.
#       """
#       self.basis_factor = basis_factor
#       try:
#           self._matrix = np.array(array, dtype=np.float64).reshape((3, 3))
#           self._matrix.setflags(write=False)
#           self._inv_matrix: Optional[np.ndarray] = None  # type
#           self._diags = None
#           self._lll_matrix_mappings: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
#           self._lll_inverse = None
#           self._a = 0
#           self._b = 0
#           self._c = 0
#           self._alpha = 0
#           self._beta = 0
#           self._gamma = 0
#       except:
#           pass
#
#
#   @property
#   def atoms_per_cell(self):
#       return self._atoms_per_cell()
#
#   def _atoms_per_cell(self):
#       """
#       Customized implementation of the calculation
#       of atoms per a cell unit.
#       """
#       raise NotImplementedError('_atoms_per_cell is not defined.')
#
#   @property
#   def primitive_lengths(self) -> Tuple[float, float, float]:
#       """
#       The lengths (a, b, c) of the primitive unit cell.
#
#       Returns
#       -------
#       Tuple of the lengths (Tuple[float, float, float])
#       """
#       return tuple(np.sqrt(np.sum(self._matrix ** 2, axis=1)).tolist())  # type: ignore
#
#   @property
#   def primitive_angles(self) -> Tuple[float, float, float]:
#       """
#       The angles (alpha, beta, gamma) of the primitive cell.
#
#       Returns
#       -------
#       Tuple of the angles (Tuple[float, float, float])
#       """
#       m = self._matrix
#       lengths = self.primitive_lengths
#       angles = np.zeros(3)
#       for i in range(3):
#           j = (i + 1) % 3
#           k = (i + 2) % 3
#           angles[i] = abs_cap(np.dot(m[j], m[k]) / (lengths[j] * lengths[k]))
#       angles = np.arccos(angles) * 180.0 / np.pi
#       return tuple(angles.tolist())  # type: ignore
#
#   @property
#   def is_orthogonal(self) -> bool:
#       """
#       Whether all primitive cell's angles are 90 degrees.
#
#       Returns
#       -------
#       Boolean (True if all angles are 90 degrees
#       """
#       return all(abs(a - 90) < 1e-5 for a in self.primitive_angles)
#
#   @property
#   def matrix(self) -> np.ndarray:
#       """
#       The lattice matrix
#
#       Returns
#       -------
#       np.array
#         The 3x3 matrix
#       """
#       return self._matrix
#
#   @property
#   def inv_matrix(self) -> np.ndarray:
#       """
#       The inverse of the lattice matrix.
#
#       Returns
#       -------
#       np.array
#         The 3x3 inverse matrix
#       """
#       if self._inv_matrix is None:
#           self._inv_matrix = np.linalg.inv(self._matrix)
#           self._inv_matrix.setflags(write=False)
#       return self._inv_matrix*(1/self.basis_factor)
#
#   @property
#   def metric_tensor(self) -> np.ndarray:
#       """
#       The metric tensor of the lattice.
#
#       Returns
#       -------
#       np.array
#         The 3x3 metric tensor of the lattice
#       """
#       return np.dot(self._matrix, self._matrix.T)
#
#   @property
#   def a(self) -> float:
#       """
#       The parameter a of the lattice.
#
#       Returns
#       -------
#       float
#       """
#       return self._a
#
#   @property
#   def b(self) -> float:
#       """
#       The parameter b of the lattice.
#
#       Returns
#       -------
#       float
#       """
#       return self._b
#
#   @property
#   def c(self) -> float:
#       """
#       The parameter c of the lattice.
#
#       Returns
#       -------
#       float
#       """
#       return self._c
#
#   @property
#   def abc(self) -> Tuple[float, float, float]:
#       """
#       Lengths of the lattice vectors, i.e. (a, b, c)
#       """
#       return (self._a, self._b, self._c)
#
#   @property
#   def alpha(self) -> float:
#       """
#       Angle alpha of lattice in degrees.
#
#       Returns
#       -------
#       float
#       """
#       return self._alpha
#
#   @property
#   def beta(self) -> float:
#       """
#       Angle beta of lattice in degrees.
#
#       Returns
#       -------
#       float
#       """
#       return self._beta
#
#   @property
#   def gamma(self) -> float:
#       """
#       Angle gamma of lattice in degrees.
#
#       Returns
#       -------
#       float
#       """
#       return self._gamma
#
#   def get_cartesian_coords(self, fractional_coords: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray]) -> np.ndarray:
#       """
#       Returns the cartesian coordinates given fractional coordinates.
#
#       Parameters
#       ----------
#       fractional_coords (3x1 array): Fractional coords.
#
#       Returns
#       -------
#       np.array
#         The Cartesian coordinates
#       """
#       return np.dot(fractional_coords, self._matrix)
#
#   def get_fractional_coords(self, cart_coords: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray]) -> np.ndarray:
#       """
#       Returns the fractional coordinates given cartesian coordinates.
#
#       Parameters
#       ----------
#       cart_coords (3x1 array): Cartesian coords.
#
#       Returns
#       -------
#       np.array
#         Fractional coordinates.
#       """
#       return np.dot(cart_coords, self.inv_matrix)
#
#   def get_vector_along_lattice_directions(self, cart_coords: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray]) -> np.ndarray:
#       """
#       Get the coordinates along the lattice directions (given cartesian coordinates).
#
#       Parameters
#       ----------
#       cart_coords (3x1 array): Cartesian coords.
#
#       Returns
#       -------
#       np.array:
#         Lattice coordinates.
#
#       Note
#       ----
#       This is different than a projection of the cartesian vector along the
#       lattice parameters. It is simply the fractional coordinates multiplied by the
#       lattice vector magnitudes.
#       """
#       return self.primitive_lengths * self.get_fractional_coords(cart_coords)
#
#   @property
#   def volume(self) -> float:
#       """
#       Volume of the unit cell.
#       """
#       m = self._matrix
#       return float(abs(np.dot(np.cross(m[0], m[1]), m[2])))
#
#
#   def construct(self):
#       """
#       Construct a lattice with the given the lattice parameters.
#
#       Returns
#       -------
#         jaqpotpy.simulation_kit.Lattice
#       """
#       return self._construct()
#
#   def _construct(self):
#       """
#       Customized implementation of the constuction of
#       the Lattice.
#       """
#       raise NotImplementedError('Constructor is not defined.')
#
#   def __call__(self, path: str, file_ext: Union[str, List[str]], **kwargs):
#     """
#     Create Bravais Lattice.
#     """
#     return self.construct()
#
#
#   def __repr__(self) -> str:
#     """
#     Convert self to repr representation.
#     Returns
#     -------
#     str
#       The string represents the class.
#     """
#     args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
#     args_names = [arg for arg in args_spec.args if arg != 'self']
#     args_info = ''
#     for arg_name in args_names:
#       value = self.__dict__[arg_name]
#       # for str
#       if isinstance(value, str):
#         value = "'" + value + "'"
#       # for list
#       if isinstance(value, list):
#         threshold = get_print_threshold()
#         value = np.array2string(np.array(value), threshold=threshold)
#       args_info += arg_name + '=' + str(value) + ', '
#     return self.__class__.__name__ + '[' + args_info[:-2] + ']'
#
#   def __str__(self) -> str:
#     """
#     Convert self to str representation.
#     Returns
#     -------
#     str
#       The string represents the class.
#     """
#     args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
#     args_names = [arg for arg in args_spec.args if arg != 'self']
#     args_num = len(args_names)
#     args_default_values = [None for _ in range(args_num)]
#     if args_spec.defaults is not None:
#       defaults = list(args_spec.defaults)
#       args_default_values[-len(defaults):] = defaults
#
#     override_args_info = ''
#     for arg_name, default in zip(args_names, args_default_values):
#       if arg_name in self.__dict__:
#         arg_value = self.__dict__[arg_name]
#         # validation
#         # skip list
#         if isinstance(arg_value, list):
#           continue
#         if isinstance(arg_value, str):
#           # skip path string
#           if "\\/." in arg_value or "/" in arg_value or '.' in arg_value:
#             continue
#         # main logic
#         if default != arg_value:
#           override_args_info += '_' + arg_name + '_' + str(arg_value)
#     return self.__class__.__name__ + override_args_info
#
#   @property
#   def reciprocal_lattice(self) -> "BravaisLattice":
#       """
#       Return the reciprocal lattice. Note that this is the standard
#       reciprocal lattice used for solid state physics with a factor of 2 *
#       pi. If you are looking for the crystallographic reciprocal lattice,
#       use the reciprocal_lattice_crystallographic property.
#       The property is lazily generated for efficiency.
#       """
#       v = np.linalg.inv(self._matrix).T
#       return BravaisLattice(v * 2 * np.pi)
#
#
# def get_print_threshold() -> int:
#   """Return the printing threshold for datasets.
#   The print threshold is the number of elements from ids/tasks to
#   print when printing representations of `Dataset` objects.
#   Returns
#   ----------
#   threshold: int
#     Number of elements that will be printed
#   """
#   return _print_threshold
#
# def abs_cap(val, max_abs_val=1):
#     """
#     Returns the value with its absolute value capped at max_abs_val.
#     Particularly useful in passing values to trignometric functions where
#     numerical errors may result in an argument > 1 being passed in.
#     Args:
#         val (float): Input value.
#         max_abs_val (float): The maximum absolute value for val. Defaults to 1.
#     Returns:
#         val if abs(val) < 1 else sign of val * max_abs_val.
#     """
#     return max(min(val, max_abs_val), -max_abs_val)