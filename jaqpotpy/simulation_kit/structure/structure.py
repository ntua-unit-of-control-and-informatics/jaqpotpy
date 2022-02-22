# import collections
# import functools
# import itertools
# import math
# import warnings
# from typing import (
#     Callable,
#     Dict,
#     Iterable,
#     List,
#     Literal,
#     Optional,
#     Sequence,
#     Tuple,
#     Union,
# )
# import numpy as np
# from jaqpotpy.simulation_kit import Bond, Composition, BravaisLattice,Element, Neighbor, PeriodicNeighbor, PeriodicSite, Site
# from jaqpotpy.simulation_kit.utils.utils import get_points_in_spheres
# 
# 
# 
# 
# class Structure(object):
#     """
#     Basic immutable Structure object with periodicity. Essentially a sequence
#     of PeriodicSites having a common BravaisLattice. IStructure is made to be
#     (somewhat) immutable so that they can function as keys in a dict. To make
#     modifications, use the standard Structure object instead. Structure
#     extends Sequence and Hashable, which means that in many cases,
#     it can be used like any Python sequence. Iterating through a
#     structure is equivalent to going through the sites in sequence.
#     """
# 
#     def __init__(
#         self,
#         lattice: BravaisLattice,
#         species: List[Union[str, Element, Composition]],
#         coords: List[Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray]],
#         charge: float = None,
#         to_unit_cell: bool = False,
#         coords_are_cartesian: bool = False,
#         site_properties: dict = None,
#     ) -> None:
#         """
#         Create a periodic structure.
#         Args:
#             BravaisLattice (BravaisLattice/3x3 array): The BravaisLattice, either as a
#                 :class:`pymatgen.core.lattice.lattice` or
#                 simply as any 2D array. Each row should correspond to a BravaisLattice
#                 vector. E.g., [[10,0,0], [20,10,0], [0,0,30]] specifies a
#                 BravaisLattice with BravaisLattice vectors [10,0,0], [20,10,0] and [0,0,30].
#             species ([Species]): Sequence of species on each site. Can take in
#                 flexible input, including:
#                 i.  A sequence of element / species specified either as string
#                     symbols, e.g. ["Li", "Fe2+", "P", ...] or atomic numbers,
#                     e.g., (3, 56, ...) or actual Element or Species objects.
#                 ii. List of dict of elements/species and occupancies, e.g.,
#                     [{"Fe" : 0.5, "Mn":0.5}, ...]. This allows the setup of
#                     disordered structures.
#             coords (Nx3 array): list of fractional/cartesian coordinates of
#                 each species.
#             charge (int): overall charge of the structure. Defaults to behavior
#                 in SiteCollection where total charge is the sum of the oxidation
#                 states.
#             validate_proximity (bool): Whether to check if there are sites
#                 that are less than 0.01 Ang apart. Defaults to False.
#             to_unit_cell (bool): Whether to map all sites into the unit cell,
#                 i.e., fractional coords between 0 and 1. Defaults to False.
#             coords_are_cartesian (bool): Set to True if you are providing
#                 coordinates in cartesian coordinates. Defaults to False.
#             site_properties (dict): Properties associated with the sites as a
#                 dict of sequences, e.g., {"magmom":[5,5,5,5]}. The sequences
#                 have to be the same length as the atomic species and
#                 fractional_coords. Defaults to None for no properties.
#         """
#         if len(species) != len(coords):
#             raise ValueError("The list of atomic species must be of the same length as the list of fractional coordinates.")
# 
#         self._lattice = BravaisLattice
# 
#         sites = []
#         for i, sp in enumerate(species):
#             prop = None
#             if site_properties:
#                 prop = {k: v[i] for k, v in site_properties.items()}
# 
#             sites.append(
#                 PeriodicSite(
#                     sp,
#                     coords[i],
#                     self._lattice,
#                     to_unit_cell,
#                     coords_are_cartesian=coords_are_cartesian,
#                     properties=prop,
#                 )
#             )
#         self._sites: Tuple[PeriodicSite, ...] = tuple(sites)
# 
#         self._charge = charge
# 
#     @classmethod
#     def from_sites(
#         cls,
#         sites: List[PeriodicSite],
#         charge: float = None,
#         validate_proximity: bool = False,
#         to_unit_cell: bool = False,
#     ) -> "Structure":
#         """
#         Convenience constructor to make a Structure from a list of sites.
#         Args:
#             sites: Sequence of PeriodicSites. Sites must have the same
#                 BravaisLattice.
#             charge: Charge of structure.
#             validate_proximity (bool): Whether to check if there are sites
#                 that are less than 0.01 Ang apart. Defaults to False.
#             to_unit_cell (bool): Whether to translate sites into the unit
#                 cell.
#         Returns:
#             (Structure) Note that missing properties are set as None.
#         """
#         if len(sites) < 1:
#             raise ValueError(f"You need at least one site to construct a {cls}")
#         prop_keys = []  # type: List[str]
#         props = {}
#         BravaisLattice = sites[0].lattice
#         for i, site in enumerate(sites):
#             if site.lattice != BravaisLattice:
#                 raise ValueError("Sites must belong to the same BravaisLattice")
#             for k, v in site.properties.items():
#                 if k not in prop_keys:
#                     prop_keys.append(k)
#                     props[k] = [None] * len(sites)
#                 props[k][i] = v
#         for k, v in props.items():
#             if any(vv is None for vv in v):
#                 warnings.warn(f"Not all sites have property {k}. Missing values are set to None.")
#         return cls(
#             BravaisLattice,
#             [site.species for site in sites],
#             [site.frac_coords for site in sites],
#             charge=charge,
#             site_properties=props,
#             validate_proximity=validate_proximity,
#             to_unit_cell=to_unit_cell,
#         )
# 
#     # @property
#     # def distance_matrix(self) -> np.ndarray:
#     #     """
#     #     Returns the distance matrix between all sites in the structure. For
#     #     periodic structures, this should return the nearest image distance.
#     #     """
#     #     return self.lattice.get_all_distances(self.frac_coords, self.frac_coords)
# 
#     @property
#     def sites(self) -> Tuple[PeriodicSite, ...]:
#         """
#         Returns an iterator for the sites in the Structure.
#         """
#         return self._sites
# 
#     @property
#     def lattice(self) -> BravaisLattice:
#         """
#         BravaisLattice of the structure.
#         """
#         return self._lattice
# 
#     # @property
#     # def density(self) -> float:
#     #     """
#     #     Returns the density in units of g/cc
#     #     """
#     #     m = Mass(self.composition.weight, "amu")
#     #     return m.to("g") / (self.volume * Length(1, "ang").to("cm") ** 3)
# 
# 
#     @property
#     def frac_coords(self):
#         """
#         Fractional coordinates as a Nx3 numpy array.
#         """
#         return np.array([site.frac_coords for site in self._sites])
# 
#     @property
#     def volume(self) -> float:
#         """
#         Returns the volume of the structure.
#         """
#         return self._lattice.volume
# 
#     def get_distance(self, i: int, j: int, jimage=None) -> float:
#         """
#         Get distance between site i and j assuming periodic boundary
#         conditions. If the index jimage of two sites atom j is not specified it
#         selects the jimage nearest to the i atom and returns the distance and
#         jimage indices in terms of BravaisLattice vector translations if the index
#         jimage of atom j is specified it returns the distance between the i
#         atom and the specified jimage atom.
#         Args:
#             i (int): Index of first site
#             j (int): Index of second site
#             jimage: Number of BravaisLattice translations in each BravaisLattice direction.
#                 Default is None for nearest image.
#         Returns:
#             distance
#         """
#         return self[i].distance(self[j], jimage)
# 
#     # def get_sites_in_sphere(
#     #     self,
#     #     pt: Union[Sequence[float], Sequence[Sequence[float]], Sequence[np.ndarray], np.ndarray],
#     #     r: float,
#     #     include_index: bool = False,
#     #     include_image: bool = False,
#     # ) -> List[PeriodicNeighbor]:
#     #     """
#     #     Find all sites within a sphere from the point, including a site (if any)
#     #     sitting on the point itself. This includes sites in other periodic
#     #     images.
#     #     Algorithm:
#     #     1. place sphere of radius r in crystal and determine minimum supercell
#     #        (parallelpiped) which would contain a sphere of radius r. for this
#     #        we need the projection of a_1 on a unit vector perpendicular
#     #        to a_2 & a_3 (i.e. the unit vector in the direction b_1) to
#     #        determine how many a_1"s it will take to contain the sphere.
#     #        Nxmax = r * length_of_b_1 / (2 Pi)
#     #     2. keep points falling within r.
#     #     Args:
#     #         pt (3x1 array): cartesian coordinates of center of sphere.
#     #         r (float): Radius of sphere.
#     #         include_index (bool): Whether the non-supercell site index
#     #             is included in the returned data
#     #         include_image (bool): Whether to include the supercell image
#     #             is included in the returned data
#     #     Returns:
#     #         [:class:`pymatgen.core.structure.PeriodicNeighbor`]
#     #     """
#     #     site_fcoords = np.mod(self.frac_coords, 1)
#     #     neighbors = []  # type: List[PeriodicNeighbor]
#     #     for fcoord, dist, i, img in self._lattice.get_points_in_sphere(site_fcoords, pt, r):
#     #         nnsite = PeriodicNeighbor(
#     #             self[i].species,
#     #             fcoord,
#     #             self._lattice,
#     #             properties=self[i].properties,
#     #             nn_distance=dist,
#     #             image=img,  # type: ignore
#     #             index=i,
#     #         )
#     #         neighbors.append(nnsite)
#     #     return neighbors
# 
#     def get_neighbors(
#         self,
#         site: PeriodicSite,
#         r: float,
#         include_index: bool = False,
#         include_image: bool = False,
#     ) -> List[PeriodicNeighbor]:
#         """
#         Get all neighbors to a site within a sphere of radius r.  Excludes the
#         site itself.
#         Args:
#             site (Site): Which is the center of the sphere.
#             r (float): Radius of sphere.
#             include_index (bool): Deprecated. Now, the non-supercell site index
#                 is always included in the returned data.
#             include_image (bool): Deprecated. Now the supercell image
#                 is always included in the returned data.
#         Returns:
#             [:class:`pymatgen.core.structure.PeriodicNeighbor`]
#         """
#         return self.get_all_neighbors(r, include_index=include_index, include_image=include_image, sites=[site])[0]
# 
#     def _get_neighbor_list_py(
#         self,
#         r: float,
#         sites: List[PeriodicSite] = None,
#         numerical_tol: float = 1e-8,
#         exclude_self: bool = True,
#     ) -> Tuple[np.ndarray, ...]:
#         """
#         A python version of getting neighbor_list. The returned values are a tuple of
#         numpy arrays (center_indices, points_indices, offset_vectors, distances).
#         Atom `center_indices[i]` has neighbor atom `points_indices[i]` that is
#         translated by `offset_vectors[i]` BravaisLattice vectors, and the distance is
#         `distances[i]`.
#         Args:
#             r (float): Radius of sphere
#             sites (list of Sites or None): sites for getting all neighbors,
#                 default is None, which means neighbors will be obtained for all
#                 sites. This is useful in the situation where you are interested
#                 only in one subspecies type, and makes it a lot faster.
#             numerical_tol (float): This is a numerical tolerance for distances.
#                 Sites which are < numerical_tol are determined to be coincident
#                 with the site. Sites which are r + numerical_tol away is deemed
#                 to be within r from the site. The default of 1e-8 should be
#                 ok in most instances.
#             exclude_self (bool): whether to exclude atom neighboring with itself within
#                 numerical tolerance distance, default to True
#         Returns: (center_indices, points_indices, offset_vectors, distances)
#         """
#         neighbors = self.get_all_neighbors_py(
#             r=r, include_index=True, include_image=True, sites=sites, numerical_tol=1e-8
#         )
#         center_indices = []
#         points_indices = []
#         offsets = []
#         distances = []
#         for i, nns in enumerate(neighbors):
#             if len(nns) > 0:
#                 for n in nns:
#                     if exclude_self and (i == n.index) and (n.nn_distance <= numerical_tol):
#                         continue
#                     center_indices.append(i)
#                     points_indices.append(n.index)
#                     offsets.append(n.image)
#                     distances.append(n.nn_distance)
#         return tuple(
#             (
#                 np.array(center_indices),
#                 np.array(points_indices),
#                 np.array(offsets),
#                 np.array(distances),
#             )
#         )
# 
#     def get_neighbor_list(
#         self,
#         r: float,
#         sites: Sequence[PeriodicSite] = None,
#         numerical_tol: float = 1e-8,
#         exclude_self: bool = True,
#     ) -> Tuple[np.ndarray, ...]:
#         """
#         Get neighbor lists using numpy array representations without constructing
#         Neighbor objects. If the cython extension is installed, this method will
#         be orders of magnitude faster than `get_all_neighbors_old` and 2-3x faster
#         than `get_all_neighbors`.
#         The returned values are a tuple of numpy arrays
#         (center_indices, points_indices, offset_vectors, distances).
#         Atom `center_indices[i]` has neighbor atom `points_indices[i]` that is
#         translated by `offset_vectors[i]` BravaisLattice vectors, and the distance is
#         `distances[i]`.
#         Args:
#             r (float): Radius of sphere
#             sites (list of Sites or None): sites for getting all neighbors,
#                 default is None, which means neighbors will be obtained for all
#                 sites. This is useful in the situation where you are interested
#                 only in one subspecies type, and makes it a lot faster.
#             numerical_tol (float): This is a numerical tolerance for distances.
#                 Sites which are < numerical_tol are determined to be coincident
#                 with the site. Sites which are r + numerical_tol away is deemed
#                 to be within r from the site. The default of 1e-8 should be
#                 ok in most instances.
#             exclude_self (bool): whether to exclude atom neighboring with itself within
#                 numerical tolerance distance, default to True
#         Returns: (center_indices, points_indices, offset_vectors, distances)
#         """
#         try:
#             from pymatgen.optimization.neighbors import (
#                 find_points_in_spheres,  # type: ignore
#             )
#         except ImportError:
#             return self._get_neighbor_list_py(r, sites, exclude_self=exclude_self)  # type: ignore
#         else:
#             if sites is None:
#                 sites = self.sites
#             site_coords = np.array([site.coords for site in sites], dtype=float)
#             cart_coords = np.ascontiguousarray(np.array(self.cart_coords), dtype=float)
#             BravaisLattice_matrix = np.ascontiguousarray(np.array(self.lattice.matrix), dtype=float)
#             r = float(r)
#             center_indices, points_indices, images, distances = find_points_in_spheres(
#                 cart_coords,
#                 site_coords,
#                 r=r,
#                 pbc=np.array([1, 1, 1], dtype=int),
#                 BravaisLattice=BravaisLattice_matrix,
#                 tol=numerical_tol,
#             )
#             cond = np.array([True] * len(center_indices))
#             if exclude_self:
#                 self_pair = (center_indices == points_indices) & (distances <= numerical_tol)
#                 cond = ~self_pair
#             return tuple(
#                 (
#                     center_indices[cond],
#                     points_indices[cond],
#                     images[cond],
#                     distances[cond],
#                 )
#             )
# 
#     def get_all_neighbors(
#         self,
#         r: float,
#         include_index: bool = False,
#         include_image: bool = False,
#         sites: Sequence[PeriodicSite] = None,
#         numerical_tol: float = 1e-8,
#     ) -> List[List[PeriodicNeighbor]]:
#         """
#         Get neighbors for each atom in the unit cell, out to a distance r
#         Returns a list of list of neighbors for each site in structure.
#         Use this method if you are planning on looping over all sites in the
#         crystal. If you only want neighbors for a particular site, use the
#         method get_neighbors as it may not have to build such a large supercell
#         However if you are looping over all sites in the crystal, this method
#         is more efficient since it only performs one pass over a large enough
#         supercell to contain all possible atoms out to a distance r.
#         The return type is a [(site, dist) ...] since most of the time,
#         subsequent processing requires the distance.
#         A note about periodic images: Before computing the neighbors, this
#         operation translates all atoms to within the unit cell (having
#         fractional coordinates within [0,1)). This means that the "image" of a
#         site does not correspond to how much it has been translates from its
#         current position, but which image of the unit cell it resides.
#         Args:
#             r (float): Radius of sphere.
#             include_index (bool): Deprecated. Now, the non-supercell site index
#                 is always included in the returned data.
#             include_image (bool): Deprecated. Now the supercell image
#                 is always included in the returned data.
#             sites (list of Sites or None): sites for getting all neighbors,
#                 default is None, which means neighbors will be obtained for all
#                 sites. This is useful in the situation where you are interested
#                 only in one subspecies type, and makes it a lot faster.
#             numerical_tol (float): This is a numerical tolerance for distances.
#                 Sites which are < numerical_tol are determined to be coincident
#                 with the site. Sites which are r + numerical_tol away is deemed
#                 to be within r from the site. The default of 1e-8 should be
#                 ok in most instances.
#         Returns:
#             [[:class:`pymatgen.core.structure.PeriodicNeighbor`], ..]
#         """
#         if sites is None:
#             sites = self.sites
#         center_indices, points_indices, images, distances = self.get_neighbor_list(
#             r=r, sites=sites, numerical_tol=numerical_tol
#         )
#         if len(points_indices) < 1:
#             return [[]] * len(sites)
#         f_coords = self.frac_coords[points_indices] + images
#         neighbor_dict: Dict[int, List] = collections.defaultdict(list)
#         BravaisLattice = self.lattice
#         atol = Site.position_atol
#         all_sites = self.sites
#         for cindex, pindex, image, f_coord, d in zip(center_indices, points_indices, images, f_coords, distances):
#             psite = all_sites[pindex]
#             csite = sites[cindex]
#             if (
#                 d > numerical_tol
#                 or
#                 # This simply compares the psite and csite. The reason why manual comparison is done is
#                 # for speed. This does not check the BravaisLattice since they are always equal. Also, the or construct
#                 # returns True immediately once one of the conditions are satisfied.
#                 psite.species != csite.species
#                 or (not np.allclose(psite.coords, csite.coords, atol=atol))
#                 or (not psite.properties == csite.properties)
#             ):
#                 neighbor_dict[cindex].append(
#                     PeriodicNeighbor(
#                         species=psite.species,
#                         coords=f_coord,
#                         BravaisLattice=BravaisLattice,
#                         properties=psite.properties,
#                         nn_distance=d,
#                         index=pindex,
#                         image=tuple(image),
#                     )
#                 )
# 
#         neighbors: List[List[PeriodicNeighbor]] = []
# 
#         for i in range(len(sites)):
#             neighbors.append(neighbor_dict[i])
#         return neighbors
# 
#     def get_all_neighbors_py(
#         self,
#         r: float,
#         include_index: bool = False,
#         include_image: bool = False,
#         sites: Sequence[PeriodicSite] = None,
#         numerical_tol: float = 1e-8,
#     ) -> List[List[PeriodicNeighbor]]:
#         """
#         Get neighbors for each atom in the unit cell, out to a distance r
#         Returns a list of list of neighbors for each site in structure.
#         Use this method if you are planning on looping over all sites in the
#         crystal. If you only want neighbors for a particular site, use the
#         method get_neighbors as it may not have to build such a large supercell
#         However if you are looping over all sites in the crystal, this method
#         is more efficient since it only performs one pass over a large enough
#         supercell to contain all possible atoms out to a distance r.
#         The return type is a [(site, dist) ...] since most of the time,
#         subsequent processing requires the distance.
#         A note about periodic images: Before computing the neighbors, this
#         operation translates all atoms to within the unit cell (having
#         fractional coordinates within [0,1)). This means that the "image" of a
#         site does not correspond to how much it has been translates from its
#         current position, but which image of the unit cell it resides.
#         Args:
#             r (float): Radius of sphere.
#             include_index (bool): Deprecated. Now, the non-supercell site index
#                 is always included in the returned data.
#             include_image (bool): Deprecated. Now the supercell image
#                 is always included in the returned data.
#             sites (list of Sites or None): sites for getting all neighbors,
#                 default is None, which means neighbors will be obtained for all
#                 sites. This is useful in the situation where you are interested
#                 only in one subspecies type, and makes it a lot faster.
#             numerical_tol (float): This is a numerical tolerance for distances.
#                 Sites which are < numerical_tol are determined to be coincident
#                 with the site. Sites which are r + numerical_tol away is deemed
#                 to be within r from the site. The default of 1e-8 should be
#                 ok in most instances.
#         Returns:
#             [[:class:`pymatgen.core.structure.PeriodicNeighbor`],...]
#         """
# 
#         if sites is None:
#             sites = self.sites
#         site_coords = np.array([site.coords for site in sites])
#         point_neighbors = get_points_in_spheres(
#             self.cart_coords,
#             site_coords,
#             r=r,
#             pbc=True,
#             numerical_tol=numerical_tol,
#             BravaisLattice=self.lattice,
#         )
#         neighbors: List[List[PeriodicNeighbor]] = []
#         for point_neighbor, site in zip(point_neighbors, sites):
#             nns: List[PeriodicNeighbor] = []
#             if len(point_neighbor) < 1:
#                 neighbors.append([])
#                 continue
#             for n in point_neighbor:
#                 coord, d, index, image = n
#                 if (d > numerical_tol) or (self[index] != site):
#                     neighbor = PeriodicNeighbor(
#                         species=self[index].species,
#                         coords=coord,
#                         lattice=self.lattice,
#                         properties=self[index].properties,
#                         nn_distance=d,
#                         index=index,
#                         image=tuple(image),
#                     )
#                     nns.append(neighbor)
#             neighbors.append(nns)
#         return neighbors
# 
#     def get_neighbors_in_shell(
#         self, origin: ArrayLike, r: float, dr: float, include_index: bool = False, include_image: bool = False
#     ) -> List[PeriodicNeighbor]:
#         """
#         Returns all sites in a shell centered on origin (coords) between radii
#         r-dr and r+dr.
#         Args:
#             origin (3x1 array): Cartesian coordinates of center of sphere.
#             r (float): Inner radius of shell.
#             dr (float): Width of shell.
#             include_index (bool): Deprecated. Now, the non-supercell site index
#                 is always included in the returned data.
#             include_image (bool): Deprecated. Now the supercell image
#                 is always included in the returned data.
#         Returns:
#             [NearestNeighbor] where Nearest Neighbor is a named tuple containing
#             (site, distance, index, image).
#         """
#         outer = self.get_sites_in_sphere(origin, r + dr, include_index=include_index, include_image=include_image)
#         inner = r - dr
#         return [t for t in outer if t.nn_distance > inner]
# 
#     def get_sorted_structure(
#         self, key: Optional[Callable] = None, reverse: bool = False
#     ) -> "Structure":
#         """
#         Get a sorted copy of the structure. The parameters have the same
#         meaning as in list.sort. By default, sites are sorted by the
#         electronegativity of the species.
#         Args:
#             key: Specifies a function of one argument that is used to extract
#                 a comparison key from each list element: key=str.lower. The
#                 default value is None (compare the elements directly).
#             reverse (bool): If set to True, then the list elements are sorted
#                 as if each comparison were reversed.
#         """
#         sites = sorted(self, key=key, reverse=reverse)
#         return self.__class__.from_sites(sites, charge=self._charge)
# 
#     def interpolate(
#         self,
#         end_structure: "Structure",
#         nimages: Union[int, Iterable] = 10,
#         interpolate_BravaisLattices: bool = False,
#         pbc: bool = True,
#         autosort_tol: float = 0,
#     ) -> List["Structure"]:
#         """
#         Interpolate between this structure and end_structure. Useful for
#         construction of NEB inputs.
#         Args:
#             end_structure (Structure): structure to interpolate between this
#                 structure and end.
#             nimages (int,list): No. of interpolation images or a list of
#                 interpolation images. Defaults to 10 images.
#             interpolate_BravaisLattices (bool): Whether to interpolate the BravaisLattices.
#                 Interpolates the lengths and angles (rather than the matrix)
#                 so orientation may be affected.
#             pbc (bool): Whether to use periodic boundary conditions to find
#                 the shortest path between endpoints.
#             autosort_tol (float): A distance tolerance in angstrom in
#                 which to automatically sort end_structure to match to the
#                 closest points in this particular structure. This is usually
#                 what you want in a NEB calculation. 0 implies no sorting.
#                 Otherwise, a 0.5 value usually works pretty well.
#         Returns:
#             List of interpolated structures. The starting and ending
#             structures included as the first and last structures respectively.
#             A total of (nimages + 1) structures are returned.
#         """
#         # Check length of structures
#         if len(self) != len(end_structure):
#             raise ValueError("Structures have different lengths!")
# 
#         if not (interpolate_BravaisLattices or self.lattice == end_structure.lattice):
#             raise ValueError("Structures with different BravaisLattices!")
# 
#         if not isinstance(nimages, collections.abc.Iterable):
#             images = np.arange(nimages + 1) / nimages
#         else:
#             images = nimages  # type: ignore
# 
#         # Check that both structures have the same species
#         for i, site in enumerate(self):
#             if site.species != end_structure[i].species:
#                 raise ValueError(
#                     "Different species!\nStructure 1:\n" + str(self) + "\nStructure 2\n" + str(end_structure)
#                 )
# 
#         start_coords = np.array(self.frac_coords)
#         end_coords = np.array(end_structure.frac_coords)
# 
#         if autosort_tol:
#             dist_matrix = self.lattice.get_all_distances(start_coords, end_coords)
#             site_mappings = collections.defaultdict(list)  # type: Dict[int, List[int]]
#             unmapped_start_ind = []
#             for i, row in enumerate(dist_matrix):
#                 ind = np.where(row < autosort_tol)[0]
#                 if len(ind) == 1:
#                     site_mappings[i].append(ind[0])
#                 else:
#                     unmapped_start_ind.append(i)
# 
#             if len(unmapped_start_ind) > 1:
#                 raise ValueError(
#                     "Unable to reliably match structures "
#                     "with auto_sort_tol = %f. unmapped indices "
#                     "= %s" % (autosort_tol, unmapped_start_ind)
#                 )
# 
#             sorted_end_coords = np.zeros_like(end_coords)
#             matched = []
#             for i, j in site_mappings.items():
#                 if len(j) > 1:
#                     raise ValueError(
#                         "Unable to reliably match structures "
#                         "with auto_sort_tol = %f. More than one "
#                         "site match!" % autosort_tol
#                     )
#                 sorted_end_coords[i] = end_coords[j[0]]
#                 matched.append(j[0])
# 
#             if len(unmapped_start_ind) == 1:
#                 i = unmapped_start_ind[0]
#                 j = list(set(range(len(start_coords))).difference(matched))[0]  # type: ignore
#                 sorted_end_coords[i] = end_coords[j]
# 
#             end_coords = sorted_end_coords
# 
#         vec = end_coords - start_coords
#         if pbc:
#             vec -= np.round(vec)
#         sp = self.species_and_occu
#         structs = []
# 
#         if interpolate_BravaisLattices:
#             # interpolate BravaisLattice matrices using polar decomposition
#             from scipy.linalg import polar
# 
#             # u is unitary (rotation), p is stretch
#             u, p = polar(np.dot(end_structure.lattice.matrix.T, np.linalg.inv(self.lattice.matrix.T)))
#             lvec = p - np.identity(3)
#             lstart = self.lattice.matrix.T
# 
#         for x in images:
#             if interpolate_BravaisLattices:
#                 l_a = np.dot(np.identity(3) + x * lvec, lstart).T
#                 lat = BravaisLattice(l_a)
#             else:
#                 lat = self.lattice
#             fcoords = start_coords + x * vec
#             structs.append(self.__class__(lat, sp, fcoords, site_properties=self.site_properties))  # type: ignore
#         return structs
# 
#     def get_miller_index_from_site_indexes(self, site_ids, round_dp=4, verbose=True):
#         """
#         Get the Miller index of a plane from a set of sites indexes.
#         A minimum of 3 sites are required. If more than 3 sites are given
#         the best plane that minimises the distance to all points will be
#         calculated.
#         Args:
#             site_ids (list of int): A list of site indexes to consider. A
#                 minimum of three site indexes are required. If more than three
#                 sites are provided, the best plane that minimises the distance
#                 to all sites will be calculated.
#             round_dp (int, optional): The number of decimal places to round the
#                 miller index to.
#             verbose (bool, optional): Whether to print warnings.
#         Returns:
#             (tuple): The Miller index.
#         """
#         return self.lattice.get_miller_index_from_coords(
#             self.frac_coords[site_ids],
#             coords_are_cartesian=False,
#             round_dp=round_dp,
#             verbose=verbose,
#         )
# 
#     def get_primitive_structure(
#         self, tolerance: float = 0.25, use_site_props: bool = False, constrain_latt: Union[List, Dict] = None
#     ):
#         """
#         This finds a smaller unit cell than the input. Sometimes it doesn"t
#         find the smallest possible one, so this method is recursively called
#         until it is unable to find a smaller cell.
#         NOTE: if the tolerance is greater than 1/2 the minimum inter-site
#         distance in the primitive cell, the algorithm will reject this BravaisLattice.
#         Args:
#             tolerance (float), Angstroms: Tolerance for each coordinate of a
#                 particular site. For example, [0.1, 0, 0.1] in cartesian
#                 coordinates will be considered to be on the same coordinates
#                 as [0, 0, 0] for a tolerance of 0.25. Defaults to 0.25.
#             use_site_props (bool): Whether to account for site properties in
#                 differentiating sites.
#             constrain_latt (list/dict): List of BravaisLattice parameters we want to
#                 preserve, e.g. ["alpha", "c"] or dict with the BravaisLattice
#                 parameter names as keys and values we want the parameters to
#                 be e.g. {"alpha": 90, "c": 2.5}.
#         Returns:
#             The most primitive structure found.
#         """
#         if constrain_latt is None:
#             constrain_latt = []
# 
#         def site_label(site):
#             if not use_site_props:
#                 return site.species_string
#             d = [site.species_string]
#             for k in sorted(site.properties.keys()):
#                 d.append(k + "=" + str(site.properties[k]))
#             return ", ".join(d)
# 
#         # group sites by species string
#         sites = sorted(self._sites, key=site_label)
# 
#         grouped_sites = [list(a[1]) for a in itertools.groupby(sites, key=site_label)]
#         grouped_fcoords = [np.array([s.frac_coords for s in g]) for g in grouped_sites]
# 
#         # min_vecs are approximate periodicities of the cell. The exact
#         # periodicities from the supercell matrices are checked against these
#         # first
#         min_fcoords = min(grouped_fcoords, key=lambda x: len(x))
#         min_vecs = min_fcoords - min_fcoords[0]
# 
#         # fractional tolerance in the supercell
#         super_ftol = np.divide(tolerance, self.lattice.abc)
#         super_ftol_2 = super_ftol * 2
# 
#         def pbc_coord_intersection(fc1, fc2, tol):
#             """
#             Returns the fractional coords in fc1 that have coordinates
#             within tolerance to some coordinate in fc2
#             """
#             d = fc1[:, None, :] - fc2[None, :, :]
#             d -= np.round(d)
#             np.abs(d, d)
#             return fc1[np.any(np.all(d < tol, axis=-1), axis=-1)]
# 
#         # here we reduce the number of min_vecs by enforcing that every
#         # vector in min_vecs approximately maps each site onto a similar site.
#         # The subsequent processing is O(fu^3 * min_vecs) = O(n^4) if we do no
#         # reduction.
#         # This reduction is O(n^3) so usually is an improvement. Using double
#         # the tolerance because both vectors are approximate
#         for g in sorted(grouped_fcoords, key=lambda x: len(x)):
#             for f in g:
#                 min_vecs = pbc_coord_intersection(min_vecs, g - f, super_ftol_2)
# 
#         def get_hnf(fu):
#             """
#             Returns all possible distinct supercell matrices given a
#             number of formula units in the supercell. Batches the matrices
#             by the values in the diagonal (for less numpy overhead).
#             Computational complexity is O(n^3), and difficult to improve.
#             Might be able to do something smart with checking combinations of a
#             and b first, though unlikely to reduce to O(n^2).
#             """
# 
#             def factors(n):
#                 for i in range(1, n + 1):
#                     if n % i == 0:
#                         yield i
# 
#             for det in factors(fu):
#                 if det == 1:
#                     continue
#                 for a in factors(det):
#                     for e in factors(det // a):
#                         g = det // a // e
#                         yield det, np.array(
#                             [
#                                 [[a, b, c], [0, e, f], [0, 0, g]]
#                                 for b, c, f in itertools.product(range(a), range(a), range(e))
#                             ]
#                         )
# 
#         # we can't let sites match to their neighbors in the supercell
#         grouped_non_nbrs = []
#         for gfcoords in grouped_fcoords:
#             fdist = gfcoords[None, :, :] - gfcoords[:, None, :]
#             fdist -= np.round(fdist)
#             np.abs(fdist, fdist)
#             non_nbrs = np.any(fdist > 2 * super_ftol[None, None, :], axis=-1)
#             # since we want sites to match to themselves
#             np.fill_diagonal(non_nbrs, True)
#             grouped_non_nbrs.append(non_nbrs)
# 
#         num_fu = functools.reduce(math.gcd, map(len, grouped_sites))
#         for size, ms in get_hnf(num_fu):
#             inv_ms = np.linalg.inv(ms)
# 
#             # find sets of BravaisLattice vectors that are are present in min_vecs
#             dist = inv_ms[:, :, None, :] - min_vecs[None, None, :, :]
#             dist -= np.round(dist)
#             np.abs(dist, dist)
#             is_close = np.all(dist < super_ftol, axis=-1)
#             any_close = np.any(is_close, axis=-1)
#             inds = np.all(any_close, axis=-1)
# 
#             for inv_m, m in zip(inv_ms[inds], ms[inds]):
#                 new_m = np.dot(inv_m, self.lattice.matrix)
#                 ftol = np.divide(tolerance, np.sqrt(np.sum(new_m**2, axis=1)))
# 
#                 valid = True
#                 new_coords = []
#                 new_sp = []
#                 new_props = collections.defaultdict(list)
#                 for gsites, gfcoords, non_nbrs in zip(grouped_sites, grouped_fcoords, grouped_non_nbrs):
#                     all_frac = np.dot(gfcoords, m)
# 
#                     # calculate grouping of equivalent sites, represented by
#                     # adjacency matrix
#                     fdist = all_frac[None, :, :] - all_frac[:, None, :]
#                     fdist = np.abs(fdist - np.round(fdist))
#                     close_in_prim = np.all(fdist < ftol[None, None, :], axis=-1)
#                     groups = np.logical_and(close_in_prim, non_nbrs)
# 
#                     # check that groups are correct
#                     if not np.all(np.sum(groups, axis=0) == size):
#                         valid = False
#                         break
# 
#                     # check that groups are all cliques
#                     for g in groups:
#                         if not np.all(groups[g][:, g]):
#                             valid = False
#                             break
#                     if not valid:
#                         break
# 
#                     # add the new sites, averaging positions
#                     added = np.zeros(len(gsites))
#                     new_fcoords = all_frac % 1
#                     for i, group in enumerate(groups):
#                         if not added[i]:
#                             added[group] = True
#                             inds = np.where(group)[0]
#                             coords = new_fcoords[inds[0]]
#                             for n, j in enumerate(inds[1:]):
#                                 offset = new_fcoords[j] - coords
#                                 coords += (offset - np.round(offset)) / (n + 2)
#                             new_sp.append(gsites[inds[0]].species)
#                             for k in gsites[inds[0]].properties:
#                                 new_props[k].append(gsites[inds[0]].properties[k])
#                             new_coords.append(coords)
# 
#                 if valid:
#                     inv_m = np.linalg.inv(m)
#                     new_l = BravaisLattice(np.dot(inv_m, self.lattice.matrix))
#                     s = Structure(
#                         new_l,
#                         new_sp,
#                         new_coords,
#                         site_properties=new_props,
#                         coords_are_cartesian=False,
#                     )
# 
#                     # Default behavior
#                     p = s.get_primitive_structure(
#                         tolerance=tolerance,
#                         use_site_props=use_site_props,
#                         constrain_latt=constrain_latt,
#                     ).get_reduced_structure()
#                     if not constrain_latt:
#                         return p
# 
#                     # Only return primitive structures that
#                     # satisfy the restriction condition
#                     p_latt, s_latt = p.lattice, self.lattice
#                     if type(constrain_latt).__name__ == "list":
#                         if all(getattr(p_latt, p) == getattr(s_latt, p) for p in constrain_latt):
#                             return p
#                     elif type(constrain_latt).__name__ == "dict":
#                         if all(getattr(p_latt, p) == constrain_latt[p] for p in constrain_latt.keys()):  # type: ignore
#                             return p
# 
#         return self.copy()
# 
#     def __repr__(self):
#         outs = ["Structure Summary", repr(self.lattice)]
#         if self._charge:
#             if self._charge >= 0:
#                 outs.append(f"Overall Charge: +{self._charge}")
#             else:
#                 outs.append(f"Overall Charge: -{self._charge}")
#         for s in self:
#             outs.append(repr(s))
#         return "\n".join(outs)
# 
#     def __str__(self):
#         outs = [
#             f"Full Formula ({self.composition.formula})",
#             f"Reduced Formula: {self.composition.reduced_formula}",
#         ]
# 
#         def to_s(x):
#             return f"{x:0.6f}"
# 
#         outs.append("abc   : " + " ".join([to_s(i).rjust(10) for i in self.lattice.abc]))
#         outs.append("angles: " + " ".join([to_s(i).rjust(10) for i in self.lattice.angles]))
#         if self._charge:
#             if self._charge >= 0:
#                 outs.append(f"Overall Charge: +{self._charge}")
#             else:
#                 outs.append(f"Overall Charge: -{self._charge}")
#         outs.append(f"Sites ({len(self)})")
#         data = []
#         props = self.site_properties
#         keys = sorted(props.keys())
#         for i, site in enumerate(self):
#             row = [str(i), site.species_string]
#             row.extend([to_s(j) for j in site.frac_coords])
#             for k in keys:
#                 row.append(props[k][i])
#             data.append(row)
#         outs.append(
#             tabulate(
#                 data,
#                 headers=["#", "SP", "a", "b", "c"] + keys,
#             )
#         )
#         return "\n".join(outs)
# 
#     def get_orderings(self, mode: Literal["enum", "sqs"] = "enum", **kwargs) -> List["Structure"]:
#         r"""
#         Returns list of orderings for a disordered structure. If structure
#         does not contain disorder, the default structure is returned.
#         Args:
#             mode ("enum" | "sqs"): Either "enum" or "sqs". If enum,
#                 the enumlib will be used to return all distinct
#                 orderings. If sqs, mcsqs will be used to return
#                 an sqs structure.
#             kwargs: kwargs passed to either
#                 pymatgen.command_line..enumlib_caller.EnumlibAdaptor
#                 or pymatgen.command_line.mcsqs_caller.run_mcsqs.
#                 For run_mcsqs, a default cluster search of 2 cluster interactions
#                 with 1NN distance and 3 cluster interactions with 2NN distance
#                 is set.
#         Returns:
#             List[Structure]
#         """
#         if self.is_ordered:
#             return [self]
#         if mode.startswith("enum"):
#             from pymatgen.command_line.enumlib_caller import EnumlibAdaptor
# 
#             adaptor = EnumlibAdaptor(self, **kwargs)
#             adaptor.run()
#             return adaptor.structures
#         if mode == "sqs":
#             from pymatgen.command_line.mcsqs_caller import run_mcsqs
# 
#             if "clusters" not in kwargs:
#                 disordered_sites = [site for site in self if not site.is_ordered]
#                 subset_structure = Structure.from_sites(disordered_sites)
#                 dist_matrix = subset_structure.distance_matrix
#                 dists = sorted(set(dist_matrix.ravel()))
#                 unique_dists = []
#                 for i in range(1, len(dists)):
#                     if dists[i] - dists[i - 1] > 0.1:
#                         unique_dists.append(dists[i])
#                 clusters = {(i + 2): d + 0.01 for i, d in enumerate(unique_dists) if i < 2}
#                 kwargs["clusters"] = clusters
#             return [run_mcsqs(self, **kwargs).bestsqs]
#         raise ValueError()
# 
#     def as_dict(self, verbosity=1, fmt=None, **kwargs):
#         """
#         Dict representation of Structure.
#         Args:
#             verbosity (int): Verbosity level. Default of 1 includes both
#                 direct and cartesian coordinates for all sites, BravaisLattice
#                 parameters, etc. Useful for reading and for insertion into a
#                 database. Set to 0 for an extremely lightweight version
#                 that only includes sufficient information to reconstruct the
#                 object.
#             fmt (str): Specifies a format for the dict. Defaults to None,
#                 which is the default format used in pymatgen. Other options
#                 include "abivars".
#             **kwargs: Allow passing of other kwargs needed for certain
#             formats, e.g., "abivars".
#         Returns:
#             JSON serializable dict representation.
#         """
#         if fmt == "abivars":
#             """Returns a dictionary with the ABINIT variables."""
#             from pymatgen.io.abinit.abiobjects import structure_to_abivars
# 
#             return structure_to_abivars(self, **kwargs)
# 
#         latt_dict = self._lattice.as_dict(verbosity=verbosity)
#         del latt_dict["@module"]
#         del latt_dict["@class"]
# 
#         d = {
#             "@module": self.__class__.__module__,
#             "@class": self.__class__.__name__,
#             "charge": self.charge,
#             "BravaisLattice": latt_dict,
#             "sites": [],
#         }
#         for site in self._sites:
#             site_dict = site.as_dict(verbosity=verbosity)
#             del site_dict["BravaisLattice"]
#             del site_dict["@module"]
#             del site_dict["@class"]
#             d["sites"].append(site_dict)
#         return d
#     #
#     # def as_dataframe(self):
#     #     """
#     #     Returns a Pandas dataframe of the sites. Structure level attributes are stored in DataFrame.attrs. Example:
#     #     Species    a    b             c    x             y             z  magmom
#     #     0    (Si)  0.0  0.0  0.000000e+00  0.0  0.000000e+00  0.000000e+00       5
#     #     1    (Si)  0.0  0.0  1.000000e-07  0.0 -2.217138e-07  3.135509e-07      -5
#     #     """
#     #     data = []
#     #     site_properties = self.site_properties
#     #     prop_keys = list(site_properties.keys())
#     #     for site in self:
#     #         row = [site.species] + list(site.frac_coords) + list(site.coords)
#     #         for k in prop_keys:
#     #             row.append(site.properties.get(k))
#     #         data.append(row)
#     #     import pandas as pd
#     #
#     #     df = pd.DataFrame(data, columns=["Species", "a", "b", "c", "x", "y", "z"] + prop_keys)
#     #     df.attrs["Reduced Formula"] = self.composition.reduced_formula
#     #     df.attrs["BravaisLattice"] = self._lattice
#     #     return df
#     #
# 
