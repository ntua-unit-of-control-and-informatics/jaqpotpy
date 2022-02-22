from pydantic import BaseModel
from typing import List, Optional


class Atoms(BaseModel):
    elements: List[str]
    coordinates: List[List[float]]
    extraInfo: Optional[list]


class Pdb(BaseModel):
    meta: dict
    atoms: Atoms

    def get_atoms(self):
        return self.atoms

class Xyz(BaseModel):
    num_atoms: int
    comment: str
    atoms: Atoms

    def get_atoms(self):
        return self.atoms


class SdfCountLine(BaseModel):
    n_atoms: int
    n_bonds: int
    n_atom_list: int
    chiral_flag: int
    n_stext_ents: int
    n_add_props: int
    version: str


class SdfBond(BaseModel):
    first_atom: int
    second_atom: int
    bond_type: int
    bond_stereo: int
    bond_topology: int
    reacting_center_status: int


class SdfHeader(BaseModel):
    title: str
    timestamp: str
    comment: str


class SdfCTable(BaseModel):
    count_line: SdfCountLine
    atoms: Atoms
    bonds: List[SdfBond]
    properties: list


class Sdf(BaseModel):
    header: SdfHeader
    con_table: SdfCTable
    extraInfo: Optional[list]

    def get_atoms(self):
        return self.con_table.atoms


