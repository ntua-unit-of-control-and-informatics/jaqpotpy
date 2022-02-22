# import numpy as np
# from jaqpotpy.simulation_kit import Composition, BravaisLattice, PeriodicSite, Site
#
#
# class Neighbor(Site):
#     """
#     Simple Site subclass to contain a neighboring atom that skips all the unnecessary checks for speed. Can be
#     used as a fixed-length tuple of size 3 to retain backwards compatibility with past use cases.
#         (site, nn_distance, index).
#     In future, usage should be to call attributes, e.g., Neighbor.index, Neighbor.distance, etc.
#     """
#
#     def __init__(
#         self,
#         species: Composition,
#         coords: np.ndarray,
#         properties: dict = None,
#         nn_distance: float = 0.0,
#         index: int = 0,
#     ):
#         """
#         :param species: Same as Site
#         :param coords: Same as Site, but must be fractional.
#         :param properties: Same as Site
#         :param nn_distance: Distance to some other Site.
#         :param index: Index within structure.
#         """
#         self.coords = coords
#         self._species = species
#         self.properties = properties or {}
#         self.nn_distance = nn_distance
#         self.index = index
#
#     def __len__(self):
#         """
#         Make neighbor Tuple-like to retain backwards compatibility.
#         """
#         return 3
#
#     def __getitem__(self, i: int):  # type: ignore
#         """
#         Make neighbor Tuple-like to retain backwards compatibility.
#         :param i:
#         :return:
#         """
#         return (self, self.nn_distance, self.index)[i]
#
#
# class PeriodicNeighbor(PeriodicSite):
#     """
#     Simple PeriodicSite subclass to contain a neighboring atom that skips all
#     the unnecessary checks for speed. Can be used as a fixed-length tuple of
#     size 4 to retain backwards compatibility with past use cases.
#         (site, distance, index, image).
#     In future, usage should be to call attributes, e.g., PeriodicNeighbor.index,
#     PeriodicNeighbor.distance, etc.
#     """
#
#     def __init__(
#         self,
#         species: Composition,
#         coords: np.ndarray,
#         lattice: BravaisLattice,
#         properties: dict = None,
#         nn_distance: float = 0.0,
#         index: int = 0,
#         image: tuple = (0, 0, 0),
#     ):
#         """
#         :param species: Same as PeriodicSite
#         :param coords: Same as PeriodicSite, but must be fractional.
#         :param lattice: Same as PeriodicSite
#         :param properties: Same as PeriodicSite
#         :param nn_distance: Distance to some other Site.
#         :param index: Index within structure.
#         :param image: PeriodicImage
#         """
#         self._lattice = lattice
#         self._frac_coords = coords
#         self._species = species
#         self.properties = properties or {}
#         self.nn_distance = nn_distance
#         self.index = index
#         self.image = image
#
#     @property  # type: ignore
#     def coords(self) -> np.ndarray:  # type: ignore
#         """
#         :return: Cartesian coords.
#         """
#         return self._lattice.get_cartesian_coords(self._frac_coords)
#
#     def __len__(self):
#         """
#         Make neighbor Tuple-like to retain backwards compatibility.
#         """
#         return 4
#
#     def __getitem__(self, i: int):  # type: ignore
#         """
#         Make neighbor Tuple-like to retain backwards compatibility.
#         :param i:
#         :return:
#         """
#         return (self, self.nn_distance, self.index, self.image)[i]
#
