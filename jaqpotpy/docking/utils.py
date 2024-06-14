"""This module adds utilities for coordinate boxes"""
from typing import List, Sequence, Tuple
import numpy as np
from scipy.spatial import ConvexHull
from typing import List, Optional, Tuple
import os
import numpy as np
from jaqpotpy.utils.types import RDKitMol
from jaqpotpy.utils.pdbqt_utils import pdbqt_to_pdb

from typing import List, Sequence, Tuple
import numpy as np
from scipy.spatial import ConvexHull


"""This module adds utilities for coordinate boxes"""
from typing import List, Sequence, Tuple
import numpy as np
from scipy.spatial import ConvexHull
from jaqpotpy.utils.rdkit_utils import load_molecule, write_molecule


def create_hydrated_pdbqt_pdb(protein_file: str, out_dir: str = "./"):
  protein_name = os.path.basename(protein_file).split(".")[0]
  protein_hyd = os.path.join(out_dir, "%s_hyd.pdb" % protein_name)
  protein_pdbqt = os.path.join(out_dir, "%s.pdbqt" % protein_name)
  protein_mol = load_molecule(protein_file,
                              calc_charges=True,
                              add_hydrogens=True)
  write_molecule(protein_mol[1], protein_hyd, is_protein=True)
  write_molecule(protein_mol[1], protein_pdbqt, is_protein=True)
  return protein_pdbqt



class CoordinateBox(object):
  """A coordinate box that represents a block in space.
  Molecular complexes are typically represented with atoms as
  coordinate points. Each complex is naturally associated with a
  number of different box regions. For example, the bounding box is a
  box that contains all atoms in the molecular complex. A binding
  pocket box is a box that focuses in on a binding region of a protein
  to a ligand. A interface box is the region in which two proteins
  have a bulk interaction.
  The `CoordinateBox` class is designed to represent such regions of
  space. It consists of the coordinates of the box, and the collection
  of atoms that live in this box alongside their coordinates.
  """

  def __init__(self, x_range: Tuple[float, float], y_range: Tuple[float, float],
               z_range: Tuple[float, float]):
    """Initialize this box.
    Parameters
    ----------
    x_range: Tuple[float, float]
      A tuple of `(x_min, x_max)` with max and min x-coordinates.
    y_range: Tuple[float, float]
      A tuple of `(y_min, y_max)` with max and min y-coordinates.
    z_range: Tuple[float, float]
      A tuple of `(z_min, z_max)` with max and min z-coordinates.
    Raises
    ------
    `ValueError` if this interval is malformed
    """
    if not isinstance(x_range, tuple) or not len(x_range) == 2:
      raise ValueError("x_range must be a tuple of length 2")
    else:
      x_min, x_max = x_range
      if not x_min <= x_max:
        raise ValueError("x minimum must be <= x maximum")
    if not isinstance(y_range, tuple) or not len(y_range) == 2:
      raise ValueError("y_range must be a tuple of length 2")
    else:
      y_min, y_max = y_range
      if not y_min <= y_max:
        raise ValueError("y minimum must be <= y maximum")
    if not isinstance(z_range, tuple) or not len(z_range) == 2:
      raise ValueError("z_range must be a tuple of length 2")
    else:
      z_min, z_max = z_range
      if not z_min <= z_max:
        raise ValueError("z minimum must be <= z maximum")
    self.x_range = x_range
    self.y_range = y_range
    self.z_range = z_range

  def __repr__(self):
    """Create a string representation of this box"""
    x_str = str(self.x_range)
    y_str = str(self.y_range)
    z_str = str(self.z_range)
    return "Box[x_bounds=%s, y_bounds=%s, z_bounds=%s]" % (x_str, y_str, z_str)

  def __str__(self):
    """Create a string representation of this box."""
    return self.__repr__()

  def __contains__(self, point: Sequence[float]) -> bool:
    """Check whether a point is in this box.
    Parameters
    ----------
    point: Sequence[float]
      3-tuple or list of length 3 or np.ndarray of shape `(3,)`.
      The `(x, y, z)` coordinates of a point in space.
    Returns
    -------
    bool
      `True` if `other` is contained in this box.
    """
    (x_min, x_max) = self.x_range
    (y_min, y_max) = self.y_range
    (z_min, z_max) = self.z_range
    x_cont = (x_min <= point[0] and point[0] <= x_max)
    y_cont = (y_min <= point[1] and point[1] <= y_max)
    z_cont = (z_min <= point[2] and point[2] <= z_max)
    return x_cont and y_cont and z_cont

  # FIXME: Argument 1 of "__eq__" is incompatible with supertype "object"
  def __eq__(self, other: "CoordinateBox") -> bool:  # type: ignore
    """Compare two boxes to see if they're equal.
    Parameters
    ----------
    other: CoordinateBox
      Compare this coordinate box to the other one.
    Returns
    -------
    bool
      That's `True` if all bounds match.
    Raises
    ------
    `ValueError` if attempting to compare to something that isn't a
    `CoordinateBox`.
    """
    if not isinstance(other, CoordinateBox):
      raise ValueError("Can only compare to another box.")
    return (self.x_range == other.x_range and self.y_range == other.y_range and
            self.z_range == other.z_range)

  def __hash__(self) -> int:
    """Implement hashing function for this box.
    Uses the default `hash` on `self.x_range, self.y_range,
    self.z_range`.
    Returns
    -------
    int
      Unique integer
    """
    return hash((self.x_range, self.y_range, self.z_range))

  def center(self) -> Tuple[float, float, float]:
    """Computes the center of this box.
    Returns
    -------
    Tuple[float, float, float]
      `(x, y, z)` the coordinates of the center of the box.
    Examples
    --------
    >>> box = CoordinateBox((0, 1), (0, 1), (0, 1))
    >>> box.center()
    (0.5, 0.5, 0.5)
    """
    x_min, x_max = self.x_range
    y_min, y_max = self.y_range
    z_min, z_max = self.z_range
    return (x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2,
            z_min + (z_max - z_min) / 2)

  def volume(self) -> float:
    """Computes and returns the volume of this box.
    Returns
    -------
    float
      The volume of this box. Can be 0 if box is empty
    Examples
    --------
    >>> box = CoordinateBox((0, 1), (0, 1), (0, 1))
    >>> box.volume()
    1
    """
    x_min, x_max = self.x_range
    y_min, y_max = self.y_range
    z_min, z_max = self.z_range
    return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

  def contains(self, other: "CoordinateBox") -> bool:
    """Test whether this box contains another.
    This method checks whether `other` is contained in this box.
    Parameters
    ----------
    other: CoordinateBox
      The box to check is contained in this box.
    Returns
    -------
    bool
      `True` if `other` is contained in this box.
    Raises
    ------
    `ValueError` if `not isinstance(other, CoordinateBox)`.
    """
    if not isinstance(other, CoordinateBox):
      raise ValueError("other must be a CoordinateBox")
    other_x_min, other_x_max = other.x_range
    other_y_min, other_y_max = other.y_range
    other_z_min, other_z_max = other.z_range
    self_x_min, self_x_max = self.x_range
    self_y_min, self_y_max = self.y_range
    self_z_min, self_z_max = self.z_range
    return (self_x_min <= other_x_min and other_x_max <= self_x_max and
            self_y_min <= other_y_min and other_y_max <= self_y_max and
            self_z_min <= other_z_min and other_z_max <= self_z_max)


def intersect_interval(interval1: Tuple[float, float],
                       interval2: Tuple[float, float]) -> Tuple[float, float]:
  """Computes the intersection of two intervals.
  Parameters
  ----------
  interval1: Tuple[float, float]
    Should be `(x1_min, x1_max)`
  interval2: Tuple[float, float]
    Should be `(x2_min, x2_max)`
  Returns
  -------
  x_intersect: Tuple[float, float]
    Should be the intersection. If the intersection is empty returns
    `(0, 0)` to represent the empty set. Otherwise is `(max(x1_min,
    x2_min), min(x1_max, x2_max))`.
  """
  x1_min, x1_max = interval1
  x2_min, x2_max = interval2
  if x1_max < x2_min:
    # If interval1 < interval2 entirely
    return (0, 0)
  elif x2_max < x1_min:
    # If interval2 < interval1 entirely
    return (0, 0)
  x_min = max(x1_min, x2_min)
  x_max = min(x1_max, x2_max)
  return (x_min, x_max)


def intersection(box1: CoordinateBox, box2: CoordinateBox) -> CoordinateBox:
  """Computes the intersection box of provided boxes.
  Parameters
  ----------
  box1: CoordinateBox
    First `CoordinateBox`
  box2: CoordinateBox
    Another `CoordinateBox` to intersect first one with.
  Returns
  -------
  CoordinateBox
    A `CoordinateBox` containing the intersection. If the intersection is empty,
    returns the box with 0 bounds.
  """
  x_intersection = intersect_interval(box1.x_range, box2.x_range)
  y_intersection = intersect_interval(box1.y_range, box2.y_range)
  z_intersection = intersect_interval(box1.z_range, box2.z_range)
  return CoordinateBox(x_intersection, y_intersection, z_intersection)


def union(box1: CoordinateBox, box2: CoordinateBox) -> CoordinateBox:
  """Merges provided boxes to find the smallest union box.
  This method merges the two provided boxes.
  Parameters
  ----------
  box1: CoordinateBox
    First box to merge in
  box2: CoordinateBox
    Second box to merge into this box
  Returns
  -------
  CoordinateBox
    Smallest `CoordinateBox` that contains both `box1` and `box2`
  """
  x_min = min(box1.x_range[0], box2.x_range[0])
  y_min = min(box1.y_range[0], box2.y_range[0])
  z_min = min(box1.z_range[0], box2.z_range[0])
  x_max = max(box1.x_range[1], box2.x_range[1])
  y_max = max(box1.y_range[1], box2.y_range[1])
  z_max = max(box1.z_range[1], box2.z_range[1])
  return CoordinateBox((x_min, x_max), (y_min, y_max), (z_min, z_max))


def merge_overlapping_boxes(boxes: List[CoordinateBox],
                            threshold: float = 0.8) -> List[CoordinateBox]:
  """Merge boxes which have an overlap greater than threshold.
  Parameters
  ----------
  boxes: list[CoordinateBox]
    A list of `CoordinateBox` objects.
  threshold: float, default 0.8
    The volume fraction of the boxes that must overlap for them to be
    merged together.
  Returns
  -------
  List[CoordinateBox]
    List[CoordinateBox] of merged boxes. This list will have length less
    than or equal to the length of `boxes`.
  """
  outputs: List[CoordinateBox] = []
  for box in boxes:
    for other in boxes:
      if box == other:
        continue
      intersect_box = intersection(box, other)
      if (intersect_box.volume() >= threshold * box.volume() or
          intersect_box.volume() >= threshold * other.volume()):
        box = union(box, other)
    unique_box = True
    for output in outputs:
      if output.contains(box):
        unique_box = False
    if unique_box:
      outputs.append(box)
  return outputs


def get_face_boxes(coords: np.ndarray, pad: float = 5.0) -> List[CoordinateBox]:
  """For each face of the convex hull, compute a coordinate box around it.
  The convex hull of a macromolecule will have a series of triangular
  faces. For each such triangular face, we construct a bounding box
  around this triangle. Think of this box as attempting to capture
  some binding interaction region whose exterior is controlled by the
  box. Note that this box will likely be a crude approximation, but
  the advantage of this technique is that it only uses simple geometry
  to provide some basic biological insight into the molecule at hand.
  The `pad` parameter is used to control the amount of padding around
  the face to be used for the coordinate box.
  Parameters
  ----------
  coords: np.ndarray
    A numpy array of shape `(N, 3)`. The coordinates of a molecule.
  pad: float, optional (default 5.0)
    The number of angstroms to pad.
  Returns
  -------
  boxes: List[CoordinateBox]
    List of `CoordinateBox`
  Examples
  --------
  >>> coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
  >>> boxes = get_face_boxes(coords, pad=5)
  """
  hull = ConvexHull(coords)
  boxes = []
  # Each triangle in the simplices is a set of 3 atoms from
  # coordinates which forms the vertices of an exterior triangle on
  # the convex hull of the macromolecule.
  for triangle in hull.simplices:
    # Points is the set of atom coordinates that make up this
    # triangular face on the convex hull
    points = np.array(
        [coords[triangle[0]], coords[triangle[1]], coords[triangle[2]]])
    # Let's extract x/y/z coords for this face
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]

    # Let's compute min/max points
    x_min, x_max = np.amin(x_coords), np.amax(x_coords)
    x_min, x_max = int(np.floor(x_min)) - pad, int(np.ceil(x_max)) + pad
    x_bounds = (x_min, x_max)

    y_min, y_max = np.amin(y_coords), np.amax(y_coords)
    y_min, y_max = int(np.floor(y_min)) - pad, int(np.ceil(y_max)) + pad
    y_bounds = (y_min, y_max)
    z_min, z_max = np.amin(z_coords), np.amax(z_coords)
    z_min, z_max = int(np.floor(z_min)) - pad, int(np.ceil(z_max)) + pad
    z_bounds = (z_min, z_max)
    box = CoordinateBox(x_bounds, y_bounds, z_bounds)
    boxes.append(box)
  return boxes


def write_vina_conf(protein_filename: str,
                    ligand_filename: str,
                    centroid: np.ndarray,
                    box_dims: np.ndarray,
                    conf_filename: str,
                    num_modes: int = 9,
                    exhaustiveness: int = None) -> None:
  """Writes Vina configuration file to disk.
  Autodock Vina accepts a configuration file which provides options
  under which Vina is invoked. This utility function writes a vina
  configuration file which directs Autodock vina to perform docking
  under the provided options.
  Parameters
  ----------
  protein_filename: str
    Filename for protein
  ligand_filename: str
    Filename for the ligand
  centroid: np.ndarray
    A numpy array with shape `(3,)` holding centroid of system
  box_dims: np.ndarray
    A numpy array of shape `(3,)` holding the size of the box to dock
  conf_filename: str
    Filename to write Autodock Vina configuration to.
  num_modes: int, optional (default 9)
    The number of binding modes Autodock Vina should find
  exhaustiveness: int, optional
    The exhaustiveness of the search to be performed by Vina
  """
  with open(conf_filename, "w") as f:
    f.write("receptor = %s\n" % protein_filename)
    f.write("ligand = %s\n\n" % ligand_filename)

    f.write("center_x = %f\n" % centroid[0])
    f.write("center_y = %f\n" % centroid[1])
    f.write("center_z = %f\n\n" % centroid[2])

    f.write("size_x = %f\n" % box_dims[0])
    f.write("size_y = %f\n" % box_dims[1])
    f.write("size_z = %f\n\n" % box_dims[2])

    f.write("num_modes = %d\n\n" % num_modes)
    if exhaustiveness is not None:
      f.write("exhaustiveness = %d\n" % exhaustiveness)


def write_gnina_conf(protein_filename: str,
                     ligand_filename: str,
                     conf_filename: str,
                     num_modes: int = 9,
                     exhaustiveness: int = None,
                     **kwargs) -> None:
  """Writes GNINA configuration file to disk.
  GNINA accepts a configuration file which provides options
  under which GNINA is invoked. This utility function writes a
  configuration file which directs GNINA to perform docking
  under the provided options.
  Parameters
  ----------
  protein_filename: str
    Filename for protein
  ligand_filename: str
    Filename for the ligand
  conf_filename: str
    Filename to write Autodock Vina configuration to.
  num_modes: int, optional (default 9)
    The number of binding modes GNINA should find
  exhaustiveness: int, optional
    The exhaustiveness of the search to be performed by GNINA
  kwargs:
    Args supported by GNINA documented here
    https://github.com/gnina/gnina#usage
  """

  with open(conf_filename, "w") as f:
    f.write("receptor = %s\n" % protein_filename)
    f.write("ligand = %s\n\n" % ligand_filename)

    f.write("autobox_ligand = %s\n\n" % protein_filename)

    if exhaustiveness is not None:
      f.write("exhaustiveness = %d\n" % exhaustiveness)
    f.write("num_modes = %d\n\n" % num_modes)

    for k, v in kwargs.items():
      f.write("%s = %s\n" % (str(k), str(v)))


def read_gnina_log(log_file: str) -> np.ndarray:
  """Read GNINA logfile and get docking scores.
  GNINA writes computed binding affinities to a logfile.
  Parameters
  ----------
  log_file: str
    Filename of logfile generated by GNINA.
  Returns
  -------
  scores: np.array, dimension (num_modes, 3)
    Array of binding affinity (kcal/mol), CNN pose score,
    and CNN affinity for each binding mode.
  """

  scores = []
  lines = open(log_file).readlines()
  mode_start = np.inf
  for idx, line in enumerate(lines):
    if line[:6] == '-----+':
      mode_start = idx
    if idx > mode_start:
      mode = line.split()
      score = [float(x) for x in mode[1:]]
      scores.append(score)

  return np.array(scores)


def load_docked_ligands(
    pdbqt_output: str) -> Tuple[List[RDKitMol], List[float]]:
  """This function loads ligands docked by autodock vina.
  Autodock vina writes outputs to disk in a PDBQT file format. This
  PDBQT file can contain multiple docked "poses". Recall that a pose
  is an energetically favorable 3D conformation of a molecule. This
  utility function reads and loads the structures for multiple poses
  from vina's output file.
  Parameters
  ----------
  pdbqt_output: str
    Should be the filename of a file generated by autodock vina's
    docking software.
  Returns
  -------
  Tuple[List[rdkit.Chem.rdchem.Mol], List[float]]
    Tuple of `molecules, scores`. `molecules` is a list of rdkit
    molecules with 3D information. `scores` is the associated vina
    score.
  Notes
  -----
  This function requires RDKit to be installed.
  """
  try:
    from rdkit import Chem
  except ModuleNotFoundError:
    raise ImportError("This function requires RDKit to be installed.")

  lines = open(pdbqt_output).readlines()
  molecule_pdbqts = []
  scores = []
  current_pdbqt: Optional[List[str]] = None
  for line in lines:
    if line[:5] == "MODEL":
      current_pdbqt = []
    elif line[:19] == "REMARK VINA RESULT:":
      words = line.split()
      # the line has format
      # REMARK VINA RESULT: score ...
      # There is only 1 such line per model so we can append it
      scores.append(float(words[3]))
    elif line[:6] == "ENDMDL":
      molecule_pdbqts.append(current_pdbqt)
      current_pdbqt = None
    else:
      # FIXME: Item "None" of "Optional[List[str]]" has no attribute "append"
      current_pdbqt.append(line)  # type: ignore

  molecules = []
  for pdbqt_data in molecule_pdbqts:
    pdb_block = pdbqt_to_pdb(pdbqt_data=pdbqt_data)
    mol = Chem.MolFromPDBBlock(str(pdb_block), sanitize=False, removeHs=False)
    molecules.append(mol)
  return molecules, scores


def prepare_inputs(protein: str,
                   ligand: str,
                   replace_nonstandard_residues: bool = True,
                   remove_heterogens: bool = True,
                   remove_water: bool = True,
                   add_hydrogens: bool = True,
                   pH: float = 7.0,
                   optimize_ligand: bool = True,
                   pdb_name: Optional[str] = None) -> Tuple[RDKitMol, RDKitMol]:
  """This prepares protein-ligand complexes for docking.
  Autodock Vina requires PDB files for proteins and ligands with
  sensible inputs. This function uses PDBFixer and RDKit to ensure
  that inputs are reasonable and ready for docking. Default values
  are given for convenience, but fixing PDB files is complicated and
  human judgement is required to produce protein structures suitable
  for docking. Always inspect the results carefully before trying to
  perform docking.
  Parameters
  ----------
  protein: str
    Filename for protein PDB file or a PDBID.
  ligand: str
    Either a filename for a ligand PDB file or a SMILES string.
  replace_nonstandard_residues: bool (default True)
    Replace nonstandard residues with standard residues.
  remove_heterogens: bool (default True)
    Removes residues that are not standard amino acids or nucleotides.
  remove_water: bool (default True)
    Remove water molecules.
  add_hydrogens: bool (default True)
    Add missing hydrogens at the protonation state given by `pH`.
  pH: float (default 7.0)
    Most common form of each residue at given `pH` value is used.
  optimize_ligand: bool (default True)
    If True, optimize ligand with RDKit. Required for SMILES inputs.
  pdb_name: Optional[str]
    If given, write sanitized protein and ligand to files called
    "pdb_name.pdb" and "ligand_pdb_name.pdb"
  Returns
  -------
  Tuple[RDKitMol, RDKitMol]
    Tuple of `protein_molecule, ligand_molecule` with 3D information.
  Note
  ----
  This function requires RDKit and OpenMM to be installed.
  Read more about PDBFixer here: https://github.com/openmm/pdbfixer.
  Examples
  --------
  >>> p, m = prepare_inputs('3cyx', 'CCC')
  >> p.GetNumAtoms()
  >> m.GetNumAtoms()
  >>> p, m = prepare_inputs('3cyx', 'CCC', remove_heterogens=False)
  >> p.GetNumAtoms()
  """

  try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from pdbfixer import PDBFixer
    from simtk.openmm.app import PDBFile
  except ModuleNotFoundError:
    raise ImportError(
        "This function requires RDKit and OpenMM to be installed.")

  if protein.endswith('.pdb'):
    fixer = PDBFixer(protein)
  else:
    fixer = PDBFixer(url='https://files.rcsb.org/download/%s.pdb' % (protein))

  if ligand.endswith('.pdb'):
    m = Chem.MolFromPDBFile(ligand)
  else:
    m = Chem.MolFromSmiles(ligand, sanitize=True)

  # Apply common fixes to PDB files
  if replace_nonstandard_residues:
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
  if remove_heterogens and not remove_water:
    fixer.removeHeterogens(True)
  if remove_heterogens and remove_water:
    fixer.removeHeterogens(False)
  if add_hydrogens:
    fixer.addMissingHydrogens(pH)

  PDBFile.writeFile(fixer.topology, fixer.positions, open('tmp.pdb', 'w'))
  p = Chem.MolFromPDBFile('tmp.pdb', sanitize=True)
  os.remove('tmp.pdb')

  # Optimize ligand
  if optimize_ligand:
    m = Chem.AddHs(m)  # need hydrogens for optimization
    AllChem.EmbedMolecule(m)
    AllChem.MMFFOptimizeMolecule(m)

  if pdb_name:
    Chem.rdmolfiles.MolToPDBFile(p, '%s.pdb' % (pdb_name))
    Chem.rdmolfiles.MolToPDBFile(m, 'ligand_%s.pdb' % (pdb_name))

  return (p, m)


class CoordinateBox(object):
  """A coordinate box that represents a block in space.
  Molecular complexes are typically represented with atoms as
  coordinate points. Each complex is naturally associated with a
  number of different box regions. For example, the bounding box is a
  box that contains all atoms in the molecular complex. A binding
  pocket box is a box that focuses in on a binding region of a protein
  to a ligand. A interface box is the region in which two proteins
  have a bulk interaction.
  The `CoordinateBox` class is designed to represent such regions of
  space. It consists of the coordinates of the box, and the collection
  of atoms that live in this box alongside their coordinates.
  """

  def __init__(self, x_range: Tuple[float, float], y_range: Tuple[float, float],
               z_range: Tuple[float, float]):
    """Initialize this box.
    Parameters
    ----------
    x_range: Tuple[float, float]
      A tuple of `(x_min, x_max)` with max and min x-coordinates.
    y_range: Tuple[float, float]
      A tuple of `(y_min, y_max)` with max and min y-coordinates.
    z_range: Tuple[float, float]
      A tuple of `(z_min, z_max)` with max and min z-coordinates.
    Raises
    ------
    `ValueError` if this interval is malformed
    """
    if not isinstance(x_range, tuple) or not len(x_range) == 2:
      raise ValueError("x_range must be a tuple of length 2")
    else:
      x_min, x_max = x_range
      if not x_min <= x_max:
        raise ValueError("x minimum must be <= x maximum")
    if not isinstance(y_range, tuple) or not len(y_range) == 2:
      raise ValueError("y_range must be a tuple of length 2")
    else:
      y_min, y_max = y_range
      if not y_min <= y_max:
        raise ValueError("y minimum must be <= y maximum")
    if not isinstance(z_range, tuple) or not len(z_range) == 2:
      raise ValueError("z_range must be a tuple of length 2")
    else:
      z_min, z_max = z_range
      if not z_min <= z_max:
        raise ValueError("z minimum must be <= z maximum")
    self.x_range = x_range
    self.y_range = y_range
    self.z_range = z_range

  def __repr__(self):
    """Create a string representation of this box"""
    x_str = str(self.x_range)
    y_str = str(self.y_range)
    z_str = str(self.z_range)
    return "Box[x_bounds=%s, y_bounds=%s, z_bounds=%s]" % (x_str, y_str, z_str)

  def __str__(self):
    """Create a string representation of this box."""
    return self.__repr__()

  def __contains__(self, point: Sequence[float]) -> bool:
    """Check whether a point is in this box.
    Parameters
    ----------
    point: Sequence[float]
      3-tuple or list of length 3 or np.ndarray of shape `(3,)`.
      The `(x, y, z)` coordinates of a point in space.
    Returns
    -------
    bool
      `True` if `other` is contained in this box.
    """
    (x_min, x_max) = self.x_range
    (y_min, y_max) = self.y_range
    (z_min, z_max) = self.z_range
    x_cont = (x_min <= point[0] and point[0] <= x_max)
    y_cont = (y_min <= point[1] and point[1] <= y_max)
    z_cont = (z_min <= point[2] and point[2] <= z_max)
    return x_cont and y_cont and z_cont

  # FIXME: Argument 1 of "__eq__" is incompatible with supertype "object"
  def __eq__(self, other: "CoordinateBox") -> bool:  # type: ignore
    """Compare two boxes to see if they're equal.
    Parameters
    ----------
    other: CoordinateBox
      Compare this coordinate box to the other one.
    Returns
    -------
    bool
      That's `True` if all bounds match.
    Raises
    ------
    `ValueError` if attempting to compare to something that isn't a
    `CoordinateBox`.
    """
    if not isinstance(other, CoordinateBox):
      raise ValueError("Can only compare to another box.")
    return (self.x_range == other.x_range and self.y_range == other.y_range and
            self.z_range == other.z_range)

  def __hash__(self) -> int:
    """Implement hashing function for this box.
    Uses the default `hash` on `self.x_range, self.y_range,
    self.z_range`.
    Returns
    -------
    int
      Unique integer
    """
    return hash((self.x_range, self.y_range, self.z_range))

  def center(self) -> Tuple[float, float, float]:
    """Computes the center of this box.
    Returns
    -------
    Tuple[float, float, float]
      `(x, y, z)` the coordinates of the center of the box.
    Examples
    --------
    >>> box = CoordinateBox((0, 1), (0, 1), (0, 1))
    >>> box.center()
    (0.5, 0.5, 0.5)
    """
    x_min, x_max = self.x_range
    y_min, y_max = self.y_range
    z_min, z_max = self.z_range
    return (x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2,
            z_min + (z_max - z_min) / 2)

  def volume(self) -> float:
    """Computes and returns the volume of this box.
    Returns
    -------
    float
      The volume of this box. Can be 0 if box is empty
    Examples
    --------
    >>> box = CoordinateBox((0, 1), (0, 1), (0, 1))
    >>> box.volume()
    1
    """
    x_min, x_max = self.x_range
    y_min, y_max = self.y_range
    z_min, z_max = self.z_range
    return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

  def contains(self, other: "CoordinateBox") -> bool:
    """Test whether this box contains another.
    This method checks whether `other` is contained in this box.
    Parameters
    ----------
    other: CoordinateBox
      The box to check is contained in this box.
    Returns
    -------
    bool
      `True` if `other` is contained in this box.
    Raises
    ------
    `ValueError` if `not isinstance(other, CoordinateBox)`.
    """
    if not isinstance(other, CoordinateBox):
      raise ValueError("other must be a CoordinateBox")
    other_x_min, other_x_max = other.x_range
    other_y_min, other_y_max = other.y_range
    other_z_min, other_z_max = other.z_range
    self_x_min, self_x_max = self.x_range
    self_y_min, self_y_max = self.y_range
    self_z_min, self_z_max = self.z_range
    return (self_x_min <= other_x_min and other_x_max <= self_x_max and
            self_y_min <= other_y_min and other_y_max <= self_y_max and
            self_z_min <= other_z_min and other_z_max <= self_z_max)


def intersect_interval(interval1: Tuple[float, float],
                       interval2: Tuple[float, float]) -> Tuple[float, float]:
  """Computes the intersection of two intervals.
  Parameters
  ----------
  interval1: Tuple[float, float]
    Should be `(x1_min, x1_max)`
  interval2: Tuple[float, float]
    Should be `(x2_min, x2_max)`
  Returns
  -------
  x_intersect: Tuple[float, float]
    Should be the intersection. If the intersection is empty returns
    `(0, 0)` to represent the empty set. Otherwise is `(max(x1_min,
    x2_min), min(x1_max, x2_max))`.
  """
  x1_min, x1_max = interval1
  x2_min, x2_max = interval2
  if x1_max < x2_min:
    # If interval1 < interval2 entirely
    return (0, 0)
  elif x2_max < x1_min:
    # If interval2 < interval1 entirely
    return (0, 0)
  x_min = max(x1_min, x2_min)
  x_max = min(x1_max, x2_max)
  return (x_min, x_max)


def intersection(box1: CoordinateBox, box2: CoordinateBox) -> CoordinateBox:
  """Computes the intersection box of provided boxes.
  Parameters
  ----------
  box1: CoordinateBox
    First `CoordinateBox`
  box2: CoordinateBox
    Another `CoordinateBox` to intersect first one with.
  Returns
  -------
  CoordinateBox
    A `CoordinateBox` containing the intersection. If the intersection is empty,
    returns the box with 0 bounds.
  """
  x_intersection = intersect_interval(box1.x_range, box2.x_range)
  y_intersection = intersect_interval(box1.y_range, box2.y_range)
  z_intersection = intersect_interval(box1.z_range, box2.z_range)
  return CoordinateBox(x_intersection, y_intersection, z_intersection)


def union(box1: CoordinateBox, box2: CoordinateBox) -> CoordinateBox:
  """Merges provided boxes to find the smallest union box.
  This method merges the two provided boxes.
  Parameters
  ----------
  box1: CoordinateBox
    First box to merge in
  box2: CoordinateBox
    Second box to merge into this box
  Returns
  -------
  CoordinateBox
    Smallest `CoordinateBox` that contains both `box1` and `box2`
  """
  x_min = min(box1.x_range[0], box2.x_range[0])
  y_min = min(box1.y_range[0], box2.y_range[0])
  z_min = min(box1.z_range[0], box2.z_range[0])
  x_max = max(box1.x_range[1], box2.x_range[1])
  y_max = max(box1.y_range[1], box2.y_range[1])
  z_max = max(box1.z_range[1], box2.z_range[1])
  return CoordinateBox((x_min, x_max), (y_min, y_max), (z_min, z_max))


def merge_overlapping_boxes(boxes: List[CoordinateBox],
                            threshold: float = 0.8) -> List[CoordinateBox]:
  """Merge boxes which have an overlap greater than threshold.
  Parameters
  ----------
  boxes: list[CoordinateBox]
    A list of `CoordinateBox` objects.
  threshold: float, default 0.8
    The volume fraction of the boxes that must overlap for them to be
    merged together.
  Returns
  -------
  List[CoordinateBox]
    List[CoordinateBox] of merged boxes. This list will have length less
    than or equal to the length of `boxes`.
  """
  outputs: List[CoordinateBox] = []
  for box in boxes:
    for other in boxes:
      if box == other:
        continue
      intersect_box = intersection(box, other)
      if (intersect_box.volume() >= threshold * box.volume() or
          intersect_box.volume() >= threshold * other.volume()):
        box = union(box, other)
    unique_box = True
    for output in outputs:
      if output.contains(box):
        unique_box = False
    if unique_box:
      outputs.append(box)
  return outputs


def get_face_boxes(coords: np.ndarray, pad: float = 5.0) -> List[CoordinateBox]:
  """For each face of the convex hull, compute a coordinate box around it.
  The convex hull of a macromolecule will have a series of triangular
  faces. For each such triangular face, we construct a bounding box
  around this triangle. Think of this box as attempting to capture
  some binding interaction region whose exterior is controlled by the
  box. Note that this box will likely be a crude approximation, but
  the advantage of this technique is that it only uses simple geometry
  to provide some basic biological insight into the molecule at hand.
  The `pad` parameter is used to control the amount of padding around
  the face to be used for the coordinate box.
  Parameters
  ----------
  coords: np.ndarray
    A numpy array of shape `(N, 3)`. The coordinates of a molecule.
  pad: float, optional (default 5.0)
    The number of angstroms to pad.
  Returns
  -------
  boxes: List[CoordinateBox]
    List of `CoordinateBox`
  Examples
  --------
  >>> coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
  >>> boxes = get_face_boxes(coords, pad=5)
  """
  hull = ConvexHull(coords)
  boxes = []
  # Each triangle in the simplices is a set of 3 atoms from
  # coordinates which forms the vertices of an exterior triangle on
  # the convex hull of the macromolecule.
  for triangle in hull.simplices:
    # Points is the set of atom coordinates that make up this
    # triangular face on the convex hull
    points = np.array(
        [coords[triangle[0]], coords[triangle[1]], coords[triangle[2]]])
    # Let's extract x/y/z coords for this face
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]

    # Let's compute min/max points
    x_min, x_max = np.amin(x_coords), np.amax(x_coords)
    x_min, x_max = int(np.floor(x_min)) - pad, int(np.ceil(x_max)) + pad
    x_bounds = (x_min, x_max)

    y_min, y_max = np.amin(y_coords), np.amax(y_coords)
    y_min, y_max = int(np.floor(y_min)) - pad, int(np.ceil(y_max)) + pad
    y_bounds = (y_min, y_max)
    z_min, z_max = np.amin(z_coords), np.amax(z_coords)
    z_min, z_max = int(np.floor(z_min)) - pad, int(np.ceil(z_max)) + pad
    z_bounds = (z_min, z_max)
    box = CoordinateBox(x_bounds, y_bounds, z_bounds)
    boxes.append(box)
  return boxes