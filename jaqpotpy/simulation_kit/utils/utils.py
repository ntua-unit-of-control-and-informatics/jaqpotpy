# from functools import reduce
# from math import gcd
# import numpy as np
# import itertools
# from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
# import math
# import collections
# from jaqpotpy.simulation_kit import BravaisLattice
#
#
# def formula_double_format(afloat, ignore_ones=True, tol=1e-8):
#     """
#     This function is used to make pretty formulas by formatting the amounts.
#     Instead of Li1.0 Fe1.0 P1.0 O4.0, you get LiFePO4.
#     Args:
#         afloat (float): a float
#         ignore_ones (bool): if true, floats of 1 are ignored.
#         tol (float): Tolerance to round to nearest int. i.e. 2.0000000001 -> 2
#     Returns:
#         A string representation of the float for formulas.
#     """
#     if ignore_ones and afloat == 1:
#         return ""
#     if abs(afloat - int(afloat)) < tol:
#         return str(int(afloat))
#     return str(round(afloat, 8))
#
# def gcd_float(lst):
#     x = reduce(gcd, lst)
#     return x
#
# def pbc_diff(fcoords1, fcoords2):
#     """
#     Returns the 'fractional distance' between two coordinates taking into
#     account periodic boundary conditions.
#     Args:
#         fcoords1: First set of fractional coordinates. e.g., [0.5, 0.6,
#             0.7] or [[1.1, 1.2, 4.3], [0.5, 0.6, 0.7]]. It can be a single
#             coord or any array of coords.
#         fcoords2: Second set of fractional coordinates.
#     Returns:
#         Fractional distance. Each coordinate must have the property that
#         abs(a) <= 0.5. Examples:
#         pbc_diff([0.1, 0.1, 0.1], [0.3, 0.5, 0.9]) = [-0.2, -0.4, 0.2]
#         pbc_diff([0.9, 0.1, 1.01], [0.3, 0.5, 0.9]) = [-0.4, -0.4, 0.11]
#     """
#     fdist = np.subtract(fcoords1, fcoords2)
#     return fdist - np.round(fdist)
#
#
# def get_points_in_spheres(
#     all_coords: np.ndarray,
#     center_coords: np.ndarray,
#     r: float,
#     pbc: Union[bool, List[bool]] = True,
#     numerical_tol: float = 1e-8,
#     lattice: BravaisLattice = None,
#     return_fcoords: bool = False,
#     ) -> List[List[Tuple[np.ndarray, float, int, np.ndarray]]]:
#     """
#     For each point in `center_coords`, get all the neighboring points in `all_coords` that are within the
#     cutoff radius `r`.
#     Args:
#         all_coords: (list of cartesian coordinates) all available points
#         center_coords: (list of cartesian coordinates) all centering points
#         r: (float) cutoff radius
#         pbc: (bool or a list of bool) whether to set periodic boundaries
#         numerical_tol: (float) numerical tolerance
#         lattice: (Lattice) lattice to consider when PBC is enabled
#         return_fcoords: (bool) whether to return fractional coords when pbc is set.
#     Returns:
#         List[List[Tuple[coords, distance, index, image]]]
#     """
#     if isinstance(pbc, bool):
#         pbc = [pbc] * 3
#     pbc = np.array(pbc, dtype=bool)  # type: ignore
#     if return_fcoords and lattice is None:
#         raise ValueError("Lattice needs to be supplied to compute fractional coordinates")
#     center_coords_min = np.min(center_coords, axis=0)
#     center_coords_max = np.max(center_coords, axis=0)
#     # The lower bound of all considered atom coords
#     global_min = center_coords_min - r - numerical_tol
#     global_max = center_coords_max + r + numerical_tol
#     if np.any(pbc):
#         if lattice is None:
#             raise ValueError("Lattice needs to be supplied when considering periodic boundary")
#         recp_len = np.array(lattice.reciprocal_lattice.abc)
#         maxr = np.ceil((r + 0.15) * recp_len / (2 * math.pi))
#         frac_coords = lattice.get_fractional_coords(center_coords)
#         nmin_temp = np.floor(np.min(frac_coords, axis=0)) - maxr
#         nmax_temp = np.ceil(np.max(frac_coords, axis=0)) + maxr
#         nmin = np.zeros_like(nmin_temp)
#         nmin[pbc] = nmin_temp[pbc]
#         nmax = np.ones_like(nmax_temp)
#         nmax[pbc] = nmax_temp[pbc]
#         all_ranges = [np.arange(x, y, dtype="int64") for x, y in zip(nmin, nmax)]
#         matrix = lattice.matrix
#         # temporarily hold the fractional coordinates
#         image_offsets = lattice.get_fractional_coords(all_coords)
#         all_fcoords = []
#         # only wrap periodic boundary
#         for k in range(3):
#             if pbc[k]:  # type: ignore
#                 all_fcoords.append(np.mod(image_offsets[:, k : k + 1], 1))
#             else:
#                 all_fcoords.append(image_offsets[:, k : k + 1])
#         all_fcoords = np.concatenate(all_fcoords, axis=1)
#         image_offsets = image_offsets - all_fcoords
#         coords_in_cell = np.dot(all_fcoords, matrix)
#         # Filter out those beyond max range
#         valid_coords = []
#         valid_images = []
#         valid_indices = []
#         for image in itertools.product(*all_ranges):
#             coords = np.dot(image, matrix) + coords_in_cell
#             valid_index_bool = np.all(
#                 np.bitwise_and(coords > global_min[None, :], coords < global_max[None, :]),
#                 axis=1,
#             )
#             ind = np.arange(len(all_coords))
#             if np.any(valid_index_bool):
#                 valid_coords.append(coords[valid_index_bool])
#                 valid_images.append(np.tile(image, [np.sum(valid_index_bool), 1]) - image_offsets[valid_index_bool])
#                 valid_indices.extend([k for k in ind if valid_index_bool[k]])
#         if len(valid_coords) < 1:
#             return [[]] * len(center_coords)
#         valid_coords = np.concatenate(valid_coords, axis=0)
#         valid_images = np.concatenate(valid_images, axis=0)
#
#     else:
#         valid_coords = all_coords  # type: ignore
#         valid_images = [[0, 0, 0]] * len(valid_coords)
#         valid_indices = np.arange(len(valid_coords))  # type: ignore
#
#     # Divide the valid 3D space into cubes and compute the cube ids
#     all_cube_index = compute_cube_index(valid_coords, global_min, r)  # type: ignore
#     nx, ny, nz = compute_cube_index(global_max, global_min, r) + 1
#     all_cube_index = three_to_one(all_cube_index, ny, nz)
#     site_cube_index = three_to_one(compute_cube_index(center_coords, global_min, r), ny, nz)
#     # create cube index to coordinates, images, and indices map
#     cube_to_coords = collections.defaultdict(list)  # type: Dict[int, List]
#     cube_to_images = collections.defaultdict(list)  # type: Dict[int, List]
#     cube_to_indices = collections.defaultdict(list)  # type: Dict[int, List]
#     for i, j, k, l in zip(all_cube_index.ravel(), valid_coords, valid_images, valid_indices):
#         cube_to_coords[i].append(j)
#         cube_to_images[i].append(k)
#         cube_to_indices[i].append(l)
#
#     # find all neighboring cubes for each atom in the lattice cell
#     site_neighbors = find_neighbors(site_cube_index, nx, ny, nz)
#     neighbors = []  # type: List[List[Tuple[np.ndarray, float, int, np.ndarray]]]
#
#     for i, j in zip(center_coords, site_neighbors):
#         l1 = np.array(three_to_one(j, ny, nz), dtype=int).ravel()
#         # use the cube index map to find the all the neighboring
#         # coords, images, and indices
#         ks = [k for k in l1 if k in cube_to_coords]
#         if not ks:
#             neighbors.append([])
#             continue
#         nn_coords = np.concatenate([cube_to_coords[k] for k in ks], axis=0)
#         nn_images = itertools.chain(*(cube_to_images[k] for k in ks))
#         nn_indices = itertools.chain(*(cube_to_indices[k] for k in ks))
#         dist = np.linalg.norm(nn_coords - i[None, :], axis=1)
#         nns: List[Tuple[np.ndarray, float, int, np.ndarray]] = []
#         for coord, index, image, d in zip(nn_coords, nn_indices, nn_images, dist):
#             # filtering out all sites that are beyond the cutoff
#             # Here there is no filtering of overlapping sites
#             if d < r + numerical_tol:
#                 if return_fcoords and (lattice is not None):
#                     coord = np.round(lattice.get_fractional_coords(coord), 10)
#                 nn = (coord, float(d), int(index), image)
#                 nns.append(nn)
#         neighbors.append(nns)
#     return neighbors
#
#
#
# # The following internal methods are used in the get_points_in_sphere method.
# def compute_cube_index(coords: np.ndarray, global_min: float, radius: float) -> np.ndarray:
#     """
#     Compute the cube index from coordinates
#     Args:
#         coords: (nx3 array) atom coordinates
#         global_min: (float) lower boundary of coordinates
#         radius: (float) cutoff radius
#     Returns: (nx3 array) int indices
#     """
#     return np.array(np.floor((coords - global_min) / radius), dtype=int)
#
#
# def one_to_three(label1d: np.ndarray, ny: int, nz: int) -> np.ndarray:
#     """
#     Convert a 1D index array to 3D index array
#     Args:
#         label1d: (array) 1D index array
#         ny: (int) number of cells in y direction
#         nz: (int) number of cells in z direction
#     Returns: (nx3) int array of index
#     """
#     last = np.mod(label1d, nz)
#     second = np.mod((label1d - last) / nz, ny)
#     first = (label1d - last - second * nz) / (ny * nz)
#     return np.concatenate([first, second, last], axis=1)
#
#
# def three_to_one(label3d: np.ndarray, ny: int, nz: int) -> np.ndarray:
#     """
#     The reverse of one_to_three
#     """
#     return np.array(label3d[:, 0] * ny * nz + label3d[:, 1] * nz + label3d[:, 2]).reshape((-1, 1))
#
# def find_neighbors(label: np.ndarray, nx: int, ny: int, nz: int) -> List[np.ndarray]:
#     """
#     Given a cube index, find the neighbor cube indices
#     Args:
#         label: (array) (n,) or (n x 3) indice array
#         nx: (int) number of cells in y direction
#         ny: (int) number of cells in y direction
#         nz: (int) number of cells in z direction
#     Returns: neighbor cell indices
#     """
#
#     array = [[-1, 0, 1]] * 3
#     neighbor_vectors = np.array(list(itertools.product(*array)), dtype=int)
#     if np.shape(label)[1] == 1:
#         label3d = one_to_three(label, ny, nz)
#     else:
#         label3d = label
#     all_labels = label3d[:, None, :] - neighbor_vectors[None, :, :]
#     filtered_labels = []
#     # filter out out-of-bound labels i.e., label < 0
#     for labels in all_labels:
#         ind = (labels[:, 0] < nx) * (labels[:, 1] < ny) * (labels[:, 2] < nz) * np.all(labels > -1e-5, axis=1)
#         filtered_labels.append(labels[ind])
#     return filtered_labels