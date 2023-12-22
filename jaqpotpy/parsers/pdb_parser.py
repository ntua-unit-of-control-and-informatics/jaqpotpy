from typing import List
from jaqpotpy.parsers.base_classes import Parser
from jaqpotpy.entities.material_models import (
    Pdb, Atoms
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


class PdbParser(Parser):
    """
    Pdb Parser.
    This class parses a pdb file (all pdb files in a folder) into a Pdb structure.
    For more information on the structure see jaqpotpy.models.material_models.Pdb

    Attributes
    ----------
    files_: str | List[str]
        File names of the pdb files that were parsed.

    References
    ----------
    .. [1] http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html

    Examples
    --------
    >>> import jaqpotpy as jt
    >>> pdb_file = './AgNP.pdb'
    >>> parser = jt.parsers.pdb_parser.pdb_parser.PdbParser()
    >>> pdb = parser.parse(pdb_file)
    >>> type(pdb)
    generator
    >>> parsed = next(pdb)
    >>> type(parsed)
    jaqpotpy.models.material_models.Pdb
    """

    @property
    def __name__(self):
        return 'PdbParser'

    def __getitem__(self):
        return self

    def _parse(self, path) -> Pdb:

        """
        Parse pdb files.

        Parameters
        ----------
        path: str
          Either the path of a certain file or a path of a folder containing
          files that will be parsed

        Returns
        -------
        jaqpotpy.models.material_models.Pdb
          A Pdb object
        """
        self.files_.append(path)

        # Initialize variables
        curr_key = ""
        pdb_dict: Pdb = Pdb(meta={}, atoms=Atoms(elements=[], coordinates=[], extraInfo=[]))
        extra = []
        lines = 0

        # Open the file and read it in as a list of rows
        with open(path) as f:
            pdb = f.read().splitlines()

        # Iterate through the file
        for row in pdb:
            curr_list = row.split()
            if curr_list[0] == "ATOM":  # Then there are specific characteristics about the atoms and we pass them to the JSON

                if curr_key != "ATOM":
                    if lines == 1:
                        pdb_dict.meta[curr_key] = extra[0]
                    else:
                        pdb_dict.meta[curr_key] = extra
                    curr_key = "ATOM"

                pdb_dict.atoms.elements.append(row[76:78].strip())
                pdb_dict.atoms.coordinates.append([
                    _str_to_num(row, 30, 38, 0), _str_to_num(row, 38, 46, 0), _str_to_num(row, 46, 54, 0)
                ])
                if row[22:26].strip() == "":
                    a = 0
                else:
                    a = _str_to_num(row, 22, 26, 1)

                pdb_dict.atoms.extraInfo.append({
                    "serial": _str_to_num(row, 6, 11, 1),
                    "name": row[12:16].strip(),
                    "altLoc": row[16],
                    "resName": row[17:20].strip(),
                    "chainID": row[21],
                    "resSeq": a,
                    "iCode": row[26],
                    "occupancy": _str_to_num(row, 54, 60, 0),
                    "tempFactor": _str_to_num(row, 60, 66, 0),
                    "charge": row[78:].strip()
                })
            else:
                # In this case we are at the begining of the pdb and we collect the meta data.

                if _clean_key(curr_list[0]) == curr_key:  # Then it is the first loop
                    extra.append(row[len(curr_key):])
                    lines += 1
                else:
                    # We check if there are multiple rocords to this specific key, and if not then we pass tha value only.
                    if curr_key != "":
                        if lines == 1:
                            pdb_dict.meta[curr_key] = extra[0]
                        else:
                            pdb_dict.meta[curr_key] = extra

                    # We change the key and initialize the dictionary and the lines variable
                    curr_key = _clean_key(curr_list[0])
                    extra = []
                    extra.append(row[len(curr_key):])
                    lines = 1
        return pdb_dict

    def _parse_dataframe(self, file: Pdb, filename: str) -> pd.DataFrame:
        """
        Parse pdb files in a pandas dataframe.

        Parameters
        ----------
        file: jaqpotpy.models.material_models.Pdb
            The Pdb structrure of a parsed file

        filename: str
            The name of the parsed file

        Returns
        -------
        pd.DataFrame()
        """

        df = pd.DataFrame()
        for i in range(len(file.atoms.extraInfo)):
            d = file.atoms.extraInfo[i]
            d['file'] = filename
            d['element'] = file.atoms.elements[i]
            d['x'] = file.atoms.coordinates[i][0]
            d['y'] = file.atoms.coordinates[i][1]
            d['z'] = file.atoms.coordinates[i][2]

            df = pd.concat([df, pd.DataFrame(d, index=[0])]).reset_index(drop=True)

        return df
