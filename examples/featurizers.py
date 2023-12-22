from jaqpotpy.descriptors import MordredDescriptors
from jaqpotpy.descriptors import RDKitDescriptors
from jaqpotpy.descriptors import TopologicalFingerprint
from jaqpotpy.descriptors import PubChemFingerprint
from jaqpotpy.descriptors import MolGraphConvFeaturizer
from jaqpotpy.descriptors import GraphData

smiles = ['COc1cccc(S(=O)(=O)Cc2cc(C(=O)NO)no2)c1', 'O=C(N[C@H](CO)[C@H](O)c1ccc([N+](=O)[O-])cc1)C(Cl)Cl'
    , 'CC(=O)NC[C@H]1CN(c2ccc3c(c2F)CCCCC3=O)C(=O)O1', 'CC(=O)NC[C@H]1CN(c2ccc(C(C)=O)cc2)C(=O)O1'
    , 'NCCNC(=O)COc1c2cccc1Cc1cccc(c1O)Cc1cccc(c1OCC(=O)NCCN)Cc1cccc(c1O)C2']



"""
Calculates Mordred descriptors

Parameters:
-ignore_3D : add 3d descriptors 
-datapoints: smiles string or rdkit mols. Descriptors are calculated for them
"""
desc = MordredDescriptors(ignore_3D=True)
f = desc.featurize_dataframe(smiles)
print(f)

"""
Calculates RDKit Descriptors

Parameters:
-datapoints: smiles string or rdkit mols. Descriptors are calculated for them
"""
rdkit_desc = RDKitDescriptors()
rd = rdkit_desc.featurize_dataframe(smiles)
print(rd)


"""
Calculates Morgan Fingerprints

Parameters:
    radius: int, optional (default 2)
      Fingerprint radius.
    size: int, optional (default 2048)
      Length of generated bit vector.
    chiral: bool, optional (default False)
      Whether to consider chirality in fingerprint generation.
    bonds: bool, optional (default True)
      Whether to consider bond order in fingerprint generation.
    features: bool, optional (default False)
      Whether to use feature information instead of atom information; see
      RDKit docs for more info.
    sparse: bool, optional (default False)
      Whether to return a dict for each molecule containing the sparse
      fingerprint.
    smiles: bool, optional (default False)
      Whether to calculate SMILES strings for fragment IDs (only applicable
      when calculating sparse fingerprints).

    -datapoints: smiles string or rdkit mols. Descriptors are calculated for them
"""
fing_desc = TopologicalFingerprint(size=1624)
fd = fing_desc.featurize(smiles)
print(fd)


"""
Calculates PubChem Fingerprints
    -datapoints: smiles string or rdkit mols. Descriptors are calculated for them
"""

pub_desc = PubChemFingerprint()
# pfp = pub_desc.featurize(smiles)
# print(pfp)

"""
Parameters
----------
use_edges: bool, default False
  Whether to use edge features or not.
use_chirality: bool, default False
  Whether to use chirality information or not.
  If True, featurization becomes slow.
use_partial_charge: bool, default False
  Whether to use partial charge data or not.
  If True, this featurizer computes gasteiger charges.
  Therefore, there is a possibility to fail to featurize for some molecules
  and featurization becomes slow.
"""

mol_graph = MolGraphConvFeaturizer(use_edges=True)
mol_g_desc = mol_graph.featurize(smiles)
print(mol_g_desc[0].num_node_features)

"""
Cast to class to get properties
"""
for g in mol_g_desc:
    graph_data = GraphData(g.node_features, g.edge_index, g.edge_features, g.node_pos_features)
    print(graph_data.num_node_features)
    print(graph_data.num_edges)
    print(graph_data.to_dgl_graph())
