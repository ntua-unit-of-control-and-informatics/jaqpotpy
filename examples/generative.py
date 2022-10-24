from rdkit import Chem
from jaqpotpy.datasets.molecular_datasets import SmilesDataset
from jaqpotpy.descriptors.molecular import MolGanFeaturizer, GraphMatrix
from jaqpotpy.models.generative.models import GanMoleculeGenerator\
    , GanMoleculeDiscriminator, MoleculeDiscriminator
from jaqpotpy.models.generative.gan import GanSolver


mols = Chem.SDMolSupplier('./data/gdb9.sdf')
max_atoms = 10
atoms_encoded = 12
smiles_gen = []
for m in mols:
    try:
        smile = Chem.MolToSmiles(m)
        smiles_gen.append(smile)
    except Exception as e:
        continue
featurizer = MolGanFeaturizer(max_atom_count=10, kekulize=True, sanitize=True)
dataset = SmilesDataset(smiles=smiles_gen[:1000], task="generation", featurizer=featurizer)
dataset.create()
generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, atoms_encoded, 0.5)
discriminator = MoleculeDiscriminator([[128, 64], 128, [128, 64]], atoms_encoded, 5 - 1, 0.5)
solver = GanSolver(generator=generator
                   , discriminator=discriminator, dataset=dataset
                   , la=1, g_lr=0.1, d_lr=0.1, batch_size=42, val_at=20, epochs=30)
solver.fit()
