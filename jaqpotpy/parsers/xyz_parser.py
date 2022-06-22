from typing import List
from jaqpotpy.parsers.base_classes import Parser
from jaqpotpy.entities.material_models import (
    Xyz, Atoms
)
import pandas as pd


def _clean_key(word) -> str:
    return ''.join(i for i in word if i.isalnum())

def _str_to_num(s, start, stop, change):
    """
    string: the whole string
    start: the starting character
    stop: the stiping character
    change: the changing function --> 0: float() and 1: int()
    """

    x = s[start:stop]

    if x.strip() != '':
        if change == 0:
            x = float(x)

        else:
            x = int(x)
    return x


class XyzParser(Parser):
    """
    XyzParser.
    This class parses a xyz or a extxyz file (or all xyz or extxyz files in a folder) into a Xyz structure.
    For more information on the structure see jaqpotpy.models.material_models.Xyz

    Attributes
    ----------
    files_: str | List[str]
        File names of the pdb files that were parsed.

    Examples
    --------
    >>> import jaqpotpy as jt
    >>> xyz_file = './AgNP.xyz'
    >>> parser = jt.parsers.xyz_parser.xyz_parser.XyzParser()
    >>> xyz = parser.parse(xyz_file)
    >>> type(xyz)
    generator
    >>> parsed = next(xyz)
    >>> type(parsed)
    jaqpotpy.models.material_models.Xyz
    """

    @property
    def __name__(self):
        return 'XyzParser'

    def __getitem__(self):
        return self

    def _parse(self, path) -> Xyz:

        """
        Parse xyz or extxyz files.

        Parameters
        ----------
        path: str
          Either the path of a certain file or a path of a folder containing
          files that will be parsed

        Returns
        -------
        jaqpotpy.models.material_models.Xyz
          A Xyz object
        """

        self.files_.append(path)

        # Initialize variables
        xyz_dict: Xyz = Xyz(num_atoms=0, comment='', atoms=Atoms(elements=[], coordinates=[], extraInfo=[]))
        cnt = 0

        # Open the file and read it in as a list of rows
        with open(path) as f:
            xyz = f.read().splitlines()

        # Iterate through the file
        for row in xyz:
            if row.strip() != '':
                if cnt == 0:
                    xyz_dict.num_atoms = int(row)
                    cnt += 1
                elif cnt == 1:
                    xyz_dict.comment = row
                    cnt += 1
                else:
                    curr_list = row.split()
                    xyz_dict.atoms.elements.append(_clean_key(curr_list[0].strip()))
                    xyz_dict.atoms.coordinates.append([float(curr_list[1]), float(curr_list[2]), float(curr_list[3])])
                    try:
                        xyz_dict.atoms.extraInfo.append(curr_list[4:])
                    except:
                        pass

        return xyz_dict

    def _parse_dataframe(self, file: Xyz, filename: str) -> pd.DataFrame:
        """
        Parse xyz files in a pandas dataframe.

        Parameters
        ----------
        file: jaqpotpy.models.material_models.Xyz
            The Xyz structrure of a parsed file

        filename: str
            The name of the parsed file

        Returns
        -------
        pd.DataFrame()
        """

        df = pd.DataFrame()
        for i in range(len(file.atoms.elements)):
            d = {}
            d['file'] = filename
            d['element'] = file.atoms.elements[i]
            d['x'] = file.atoms.coordinates[i][0]
            d['y'] = file.atoms.coordinates[i][1]
            d['z'] = file.atoms.coordinates[i][2]

            df = pd.concat([df, pd.DataFrame(d, index=[0])]).reset_index(drop=True)

        return df
