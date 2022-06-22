from typing import List
from jaqpotpy.parsers.base_classes import Parser
from jaqpotpy.entities.material_models import (
    Atoms, Sdf, SdfHeader, SdfCTable, SdfBond, SdfCountLine
)
import pandas as pd
from pydantic import parse_obj_as


def _clean_key(word) -> str:
    return ''.join(i for i in word if i.isalnum())


class MolParser(Parser):
    """
    MolParser.
    This class parses a mol or sdf file (or all mol or sdf files in a folder) into a Sdf structure.
    For more information on the structure see jaqpotpy.models.material_models.Sdf

    Attributes
    ----------
    files_: str | List[str]
        File names of the pdb files that were parsed.

    Note
    ----

    Examples
    --------
    >>> import jaqpotpy as jt
    >>> mol_file = './caffeine.mol'
    >>> parser = jt.parsers.mol_parser.mol_parser.MolParser()
    >>> mol = parser.parse(mol_file)
    >>> type(mol)
    generator
    >>> parsed = next(mol)
    >>> type(parsed)
    jaqpotpy.models.material_models.Sdf
    """

    @property
    def __name__(self):
        return 'MolParser'


    def __getitem__(self):
        return self

    def _parse(self, path:str) -> Sdf:

        """
        Parse sdf and mol files.

        Parameters
        ----------
        path: str
          Either the path of a certain file or a path of a folder containing
          files that will be parsed

        Returns
        -------
        jaqpotpy.models.material_models.Sdf
          A Sdf object
        """
        # Initialize variables
        sdf_dict: Sdf = Sdf(header=SdfHeader(title='', timestamp='', comment=''),
                            con_table=SdfCTable(
                                count_line=SdfCountLine(n_atoms=0, n_bonds=0, n_atom_list=0, chiral_flag=0, n_stext_ents=0, n_add_props=0, version=''),
                                atoms=Atoms(elements=[], coordinates=[], extraInfo=[]),
                                bonds=[SdfBond(first_atom=0, second_atom=0, bond_type=0, bond_stereo=0, bond_topology=0, reacting_center_status=0)],
                                properties=[]),
                            extraInfo=[])

        head: SdfHeader = SdfHeader(title='', timestamp='', comment='')
        ctable: SdfCTable = SdfCTable(
            count_line=SdfCountLine(n_atoms=0, n_bonds=0, n_atom_list=0, chiral_flag=0, n_stext_ents=0, n_add_props=0, version=''),
            atoms=Atoms(elements=[], coordinates=[], extraInfo=[]),
            bonds=[SdfBond(first_atom=0, second_atom=0, bond_type=0, bond_stereo=0, bond_topology=0, reacting_center_status=0)],
            properties=[])
        records = []

        # Open the file and read it in as a list of rows
        with open(path) as f:
            sdf = f.read().splitlines()

        # Iterate through the file
        line_cnt = 0
        extraProp = ''
        cont = True

        for row in sdf:
            if line_cnt == 0:
                head.title = row
            elif line_cnt == 1:
                head.timestamp = row
            elif line_cnt == 2:
                head.comment = row
            elif line_cnt == 3:
                curr_list = row.split()
                try:
                    ctable.count_line.n_atoms = int(curr_list[0])
                    ctable.count_line.n_bonds = int(curr_list[1])
                    ctable.count_line.n_atom_list = int(curr_list[2])
                    ctable.count_line.chiral_flag = int(curr_list[3])
                    ctable.count_line.n_stext_ents = int(curr_list[4])
                    ctable.count_line.n_add_props = int(curr_list[5])
                    ctable.count_line.version = curr_list[6]
                except:
                    raise TypeError('Count line has wrong format. Count Line: ', row)
            else:
                curr_list = row.split()
                try:
                    if cont:
                        _ = int(curr_list[0])
                    else:
                        _ = int('')
                except ValueError:
                    if row.strip() != '':
                        if extraProp != '':
                            sdf_dict.extraInfo.append(extraProp + row)
                            extraProp = ''
                        elif curr_list[0].strip().lower() == 'm':
                            if curr_list[1].strip().lower() != 'end':
                                ctable.properties.append(row)
                            else:
                                cont = False
                                sdf_dict.header = head
                                sdf_dict.con_table = ctable
                        elif curr_list[0].strip() == '>':
                            if extraProp == '':
                                extraProp += row[1:][row[1:].find('<') + 1: row[1:].find('>')] + ': '

                        elif row[0] == '$':
                            records.append(sdf_dict)
                            self.files_.append(path)
                            sdf_dict: Sdf = Sdf(header=SdfHeader(title='', timestamp='', comment=''),
                                                con_table=SdfCTable(
                                                    count_line=SdfCountLine(n_atoms=0, n_bonds=0, n_atom_list=0,
                                                                            chiral_flag=0, n_stext_ents=0,
                                                                            n_add_props=0, version=''),
                                                    atoms=Atoms(elements=[], coordinates=[], extraInfo=[]),
                                                    bonds=[
                                                        SdfBond(first_atom=0, second_atom=0, bond_type=0, bond_stereo=0,
                                                                bond_topology=0, reacting_center_status=0)],
                                                    properties=[]),
                                                extraInfo=[])

                            head: SdfHeader = SdfHeader(title='', timestamp='', comment='')
                            ctable: SdfCTable = SdfCTable(
                                count_line=SdfCountLine(n_atoms=0, n_bonds=0, n_atom_list=0, chiral_flag=0,
                                                        n_stext_ents=0, n_add_props=0, version=''),
                                atoms=Atoms(elements=[], coordinates=[], extraInfo=[]),
                                bonds=[SdfBond(first_atom=0, second_atom=0, bond_type=0, bond_stereo=0, bond_topology=0,
                                               reacting_center_status=0)],
                                properties=[]
                            )
                            line_cnt = -1
                        else:
                            try:
                                ctable.atoms.elements.append(_clean_key(curr_list[3]))
                                ctable.atoms.coordinates.append(
                                    [float(curr_list[0]), float(curr_list[1]), float(curr_list[2])])
                                ctable.atoms.extraInfo.append(curr_list[4:])
                            except:
                                # empty row
                                pass
                except IndexError:
                    pass
                except Exception as e:
                    raise TypeError('File has malformed fields. Error: ', str(e))
                else:
                    try:
                        ctable.bonds.append(parse_obj_as(SdfBond, {
                            'first_atom': int(curr_list[0]),
                            'second_atom': int(curr_list[1]),
                            'bond_type': int(curr_list[2]),
                            'bond_stereo': int(curr_list[3]),
                            'bond_topology': int(curr_list[4]),
                            'reacting_center_status': int(curr_list[5]),

                        }))
                    except:
                        raise TypeError('Found bond with wrong format. Bond: ', row)

            line_cnt += 1

        if len(records) > 1:
            return records

        try:
            return records[0]
        except:
            return sdf_dict

    def _parse_dataframe(self, file: Sdf, filename: str) -> List[pd.DataFrame]:
        """
        Parse mol or sdf files in a pandas dataframe.

         Parameters
        ----------
        file: jaqpotpy.models.material_models.Sdf
            The Sdf structrure of a parsed file

        filename: str
            The name of the parsed file

        Returns
        -------
        List[pd.DataFrame()]
        """
        atoms = pd.DataFrame()
        bonds = pd.DataFrame()

        for i in range(len(file.con_table.atoms.elements)):
            d = {}
            d['file'] = filename
            d['element'] = file.con_table.atoms.elements[i]
            d['x'] = file.con_table.atoms.coordinates[i][0]
            d['y'] = file.con_table.atoms.coordinates[i][1]
            d['z'] = file.con_table.atoms.coordinates[i][2]

            atoms = pd.concat([atoms, pd.DataFrame(d, index=[0])]).reset_index(drop=True)

        for bond in file.con_table.bonds:
            d = bond.dict()
            d['file'] = filename
            bonds = pd.concat([bonds, pd.DataFrame(d, index=[0])]).reset_index(drop=True)

        return [atoms, bonds]
