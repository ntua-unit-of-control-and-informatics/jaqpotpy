from typing import Union, List, Optional, Any
import numpy as np
from scipy.spatial import Delaunay
from jaqpotpy.helpers.periodic_table.element import Element
from tqdm import tqdm
import warnings
import pandas as pd
from jaqpotpy.descriptors.base_classes import MaterialFeaturizer
from jaqpotpy.cfg import config
from jaqpotpy.entities.material_models import *
from jaqpotpy.parsers import *

class GeomDescriptors(MaterialFeaturizer):
    """Geometrical Descriptors.
      This class computes a list of of geometrical descriptors of materials based on the Delaunay tessellation and 14 physicochemical properties of the involved atoms.
        Namely:
            - Atomic Number
            - Atomic radius
            - Atomic Weight
            - Atomic Volume
            - Boiling Point
            - Density
            - Dipole Polarizability
            - Evaporation Heat
            - Fusion Heat
            - Radioactivity (0 for non radioactive atoms, 1 for radioactive atoms)
            - Lattice Constant
            - Thermal Conductivity
            - Specific Heat
            - Electronegativity (Pauling scale)

      Attributes
      ----------
      descriptors: Optional[list]
        List of descriptor names used in this class.
        Items of the list must be selected from the set:
            "atomic_number" | "atomic_radius" | "atomic_weight" | "atomic_volume" |
            "boiling_point" | "density" | "dipole_polarizability" | "evaporation_heat" |
            "fusion_heat" | "is_radioactive" | "lattice_constant" | "thermal_conductivity" |
            "specific_heat" | "en_pauling"

      Examples
      --------
      >>> import jaqpotpy as jt
      >>> pdb_file = './AgNP.pdb'
      >>> featurizer = jt.descriptors.material.GeomDescriptors()
      >>> features = featurizer.featurize(pdb_file)
      >>> type(features[0])
      <class 'numpy.ndarray'>
    """

    @property
    def __name__(self):
        return 'GeomDescriptors'

    def __init__(self, descriptors: List[str] = None):
        if descriptors:
            for d in descriptors:
                if d not in ["atomic_number", "atomic_radius", "atomic_weight", "atomic_volume", "boiling_point", "density", "dipole_polarizability",
                             "evaporation_heat", "fusion_heat", "is_radioactive", "lattice_constant", "thermal_conductivity", "specific_heat", "en_pauling"]:
                    raise ValueError('Descriptor {} is not supported from this Featurizer. Consider building your custom featurizer'.format(d))
            self.descriptors = descriptors
        else:
            self.descriptors = ["atomic_number", "atomic_radius", "atomic_weight", "atomic_volume", "boiling_point", "density", "dipole_polarizability",
                             "evaporation_heat", "fusion_heat", "is_radioactive", "lattice_constant", "thermal_conductivity", "specific_heat", "en_pauling"]

        self.data = pd.DataFrame()

    def __getitem__(self):
        return self

    def _featurize(self, material: Union[str, Pdb, Xyz, Sdf], **kwargs) -> np.ndarray:
        """
        Calculate Geometrical Descriptors.
          This class computes a list of of geometrical descriptors of materials based on the Delaunay tessellation and 14 physicochemical properties of the involved atoms.
            Namely:
                - Atomic Number
                - Atomic radius
                - Atomic Weight
                - Atomic Volume
                - Boiling Point
                - Density
                - Dipole Polarizability
                - Evaporation Heat
                - Fusion Heat
                - Radioactivity (0 for non radioactive atoms, 1 for radioactive atoms)
                - Lattice Constant
                - Thermal Conductivity
                - Specific Heat
                - Electronegativity (Pauling scale)
        """
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

        ret_dataframe = pd.DataFrame()

        if config.verbose is False:
            disable_tq = True
        else:
            disable_tq = False

        if isinstance(material, str):
            file_ext = material.split('.')[-1].lower()
            material_list = []
            if file_ext in ['sdf', 'mol']:
                parser = MolParser(material, file_ext)
            elif file_ext in ['pdb']:
                parser = PdbParser(material, file_ext)
            elif file_ext in ['xyz', 'extxyz']:
                parser = XyzParser(material, file_ext)
            else:
                raise ValueError(f'Files with extention {file_ext} are not supported. Supported file formats are "xyz","extxyz","pdb","sdf" and "mol".')

            for mat in parser.parse():
                material_list.append(mat)
        else:
            material_list = [material]


        for i, item in enumerate(tqdm(material_list, desc='Parsing files', disable=disable_tq)):
            obj = item.get_atoms()

            points = obj.coordinates
            atoms = obj.elements

            # Creation of tetrahedra of neighboring atoms.
            delaunay_tess = Delaunay(points)
            tetrahedra = delaunay_tess.simplices

            # Initialization of local variables
            freq = {}
            new_line = pd.DataFrame(index=[0])

            # Loop through all tetrahedra
            for tetrahedron in tetrahedra:

                # Create the tetrahedron key (e.g. Tetrahedron for elements C,H,N,O -> descriptor = CHNO)
                lst = [atoms[i].upper().strip() for i in tetrahedron]
                descriptor = ''.join(sorted(lst))

                # If the tetrahedron has already been found, increase it's frequency.
                if descriptor in [col[:col.find('_')] for col in new_line.columns]:
                    freq[descriptor] += 1

                # If the tetrahedron has not been found, calculate the total physicochemical properties of the tetrahedron and store add them to the DataFrame.
                else:
                    # Initialization
                    an = 0  # Atomic Number
                    ar = 0  # Atomic Radius
                    aw = 0  # Atomic Weight
                    av = 0  # Atomic Volume
                    bp = 0  # Boiling Point
                    de = 0  # Density
                    dp = 0  # Dipole Polarizability
                    eh = 0  # Evaporation Heat
                    fh = 0  # Fusion Heat
                    ra = 0  # Radioactivity
                    lc = 0  # Lattice Constant
                    tc = 0  # Thermal Conductivity
                    sh = 0  # Specific Heat
                    en = 0  # Electronegativity (Pauling scale)

                    # Loop through all atoms of the tetrahedron and add the according properties.
                    for i in lst:
                        el = Element(correct_element_format(i))
                        elmnt = el.parameters
                        an += elmnt.atomic_number
                        ar += elmnt.atomic_radius
                        aw += elmnt.atomic_weight
                        av += elmnt.atomic_volume
                        bp += elmnt.boiling_point
                        de += elmnt.density
                        if elmnt.dipole_polarizability:
                            dp += elmnt.dipole_polarizability
                        if elmnt.evaporation_heat:
                            eh += elmnt.evaporation_heat
                        if elmnt.fusion_heat:
                            fh += elmnt.fusion_heat
                        if elmnt.is_radioactive:
                            ra += elmnt.is_radioactive.bit_length()
                        if elmnt.lattice_constant:
                            lc += elmnt.lattice_constant
                        if elmnt.thermal_conductivity:
                            tc += elmnt.thermal_conductivity
                        if elmnt.specific_heat:
                            sh += elmnt.specific_heat
                        if elmnt.en_pauling:
                            en += elmnt.en_pauling

                    guide = {
                        "atomic_number": an,
                        "atomic_radius": ar,
                        "atomic_weight": aw,
                        "atomic_volume": av,
                        "boiling_point": bp,
                        "density": de,
                        "dipole_polarizability": dp,
                        "evaporation_heat": eh,
                        "fusion_heat": fh,
                        "is_radioactive": ra,
                        "lattice_constant": lc,
                        "thermal_conductivity": tc,
                        "specific_heat": sh,
                        "en_pauling": en
                    }

                    # Store the total physicochemical properties to the DataFrame
                    for d in self.descriptors:
                        new_line[descriptor + '_' + d] = guide[d]

                    # Set the found counter equal to 1.
                    freq[descriptor] = 1

            # Get the weighted average of the total physicochemical properties according to the frequency of the tetrahedra.
            for key, value in freq.items():
                value /= len(tetrahedra)
                for i in [col for col in new_line.columns if col[:4] == key]:
                    new_line[i] *= value

            ret_dataframe = pd.concat([ret_dataframe, new_line]).reset_index(drop=True)

        self.data = ret_dataframe

        return np.array(self.data)

    def _get_column_names(self, **kwargs) -> list:
        """
        Return the column names
        """
        return list(self.data.columns)

    def _featurize_dataframe(self, material, **kwargs) -> pd.DataFrame:
        """
        Calculate Mordred descriptors.
        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          1D array of Mordred descriptors for `mol`.
          If ignore_3D is True, the length is 1613.
          If ignore_3D is False, the length is 1826.
        """
        _ = self._featurize(material)
        return self.data


def correct_element_format(element):
    """
    Correction of an elements format.

    Examples:
        AU -> Au
        C  -> C
        PB -> Pb

    element : str
        The element's name.
    """
    return ''.join([element[i] if i == 0 else element[i].lower() for i in range(len(element))])

