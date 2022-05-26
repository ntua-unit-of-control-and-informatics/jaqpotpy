from jaqpotpy.utils.types import RDKitMol


ZINC_SMILES_CHARSET = ["C", "N", "O", "F", "B", "I", "H", "S", "P", "Cl", "Br"]


def add_H(mol: RDKitMol):
    mol.add_atom(1)
    return mol


def remove_H(mol: RDKitMol):
    mol.remove_atom(1)
    return mol


def add_B(mol: RDKitMol):
    mol.add_atom(5)
    return mol


def remove_B(mol: RDKitMol):
    mol.remove_atom(5)
    return mol


def add_I(mol: RDKitMol):
    mol.add_atom(53)
    return mol


def remove_I(mol: RDKitMol):
    mol.remove_atom(53)
    return mol


def add_C(mol: RDKitMol):
    mol.add_atom(6)
    return mol


def remove_C(mol: RDKitMol):
    mol.remove_atom(6)
    return mol


def add_N(mol: RDKitMol):
    mol.add_atom(7)
    return mol


def remove_N(mol: RDKitMol):
    mol.remove_atom(7)
    return mol


def add_O(mol: RDKitMol):
    mol.add_atom(8)
    return mol


def remove_O(mol: RDKitMol):
    mol.remove_atom(8)
    return mol


def add_F(mol: RDKitMol):
    mol.add_atom(9)
    return mol


def remove_F(mol: RDKitMol):
    mol.remove_atom(9)
    return mol


def add_P(mol: RDKitMol):
    mol.add_atom(15)
    return mol


def remove_P(mol: RDKitMol):
    mol.remove_atom(15)
    return mol


def add_S(mol: RDKitMol):
    mol.add_atom(16)
    return mol


def remove_S(mol: RDKitMol):
    mol.remove_atom(16)
    return mol


def add_Cl(mol: RDKitMol):
    mol.add_atom(17)
    return mol


def remove_Cl(mol: RDKitMol):
    mol.remove_atom(17)
    return mol


def add_Br(mol: RDKitMol):
    mol.add_atom(35)
    return mol


def remove_Br(mol: RDKitMol):
    mol.remove_atom(35)
    return mol
