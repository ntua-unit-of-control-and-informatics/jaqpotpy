import unittest
from jaqpotpy.datasets.molecular_datasets import SmilesDataset
from jaqpotpy.descriptors.molecular import MolGanFeaturizer, GraphMatrix
from jaqpotpy.models.generative.models import GanMoleculeGenerator\
    , GanMoleculeDiscriminator, MoleculeDiscriminator
from jaqpotpy.models.generative.gan import GanSolver
from torch.utils.data import DataLoader
import torch.nn.functional as F
from rdkit import Chem
import torch


class TestModels(unittest.TestCase):

    def setUp(self):
        from tdc.generation import MolGen
        try:
            self.smiles = []
            with open("./data/zinc.tab", "r") as f:
                for l in f.readlines():
                    if l != "smiles\n":
                        self.smiles.append(l.replace('\n', "").replace("\"", ""))
        except FileNotFoundError:
            self.data = MolGen(name='ZINC')
            self.smiles = self.data.__dict__['smiles_lst']

    def test_smiles_parser(self):
        for s in self.smiles[0:10080]:
            print(s)
            mol = Chem.MolFromSmiles(s)
            feat = MolGanFeaturizer(max_atom_count=42)
            f = feat.featurize(s)
            print(f)
            print(mol)

    def test_qm9(self):

        mols = Chem.SDMolSupplier('./data/gdb9.sdf')
        smiles_dif = []
        for m in mols:
            try:
                sm = Chem.MolToSmiles(m)
                print("SMILES FOR FEAT: " + Chem.MolToSmiles(m))
                featurizer = MolGanFeaturizer(max_atom_count=12, kekulize=True, sanitize=False)
                mf = featurizer.featurize(Chem.MolToSmiles(m))
                mfde = featurizer.defeaturize(mf)

                if mfde:
                    sm_c = Chem.MolToSmiles(mfde[0])
                    if sm != sm_c:
                        smiles_dif.append(sm)
                    print("SMILES DEFEAT: " + Chem.MolToSmiles(mfde[0]))
                print(smiles_dif)
            except Exception as e:
                pass

    def test_two_graphs(self):
        featurizer = MolGanFeaturizer(max_atom_count=12, kekulize=True, sanitize=False)
        smile_1 = "CC(=O)OCN"
        print("Featurizing smile 1: " + smile_1)
        g = featurizer.featurize(smile_1)
        g_def = featurizer.defeaturize(g)
        smile_2 = Chem.MolToSmiles(g_def[0])
        print("Deafeturized smile 1 : " + smile_2)

        smile_3 = "C[NH+]1CC(O)C1"
        g_2 = featurizer.featurize(smile_3)
        print("COMPARING GRAPHS")
        self.assertEqual(g, g)

    def test_smiles_list(self):
        featurizer = MolGanFeaturizer(max_atom_count=12, kekulize=True, sanitize=False)
        smiles = ['[CH]C#C[CH]', '[CH]C#C[NH3+]', 'C[NH+]1CC1', 'COC(C)[NH-]', '[NH-]C1CCO1', 'CC1C[NH+]1C', '[NH-][CH+]OC=O',
         'C[NH2+][CH+]OC', 'CC[NH+]1CC1', 'C#C/C(C)=N/O', 'C/C(CO)=N\\O', 'CC/C(C)=N/O', '[NH-]c1[nH+]cco1',
         '[NH-]C1OC=CO1', 'CC([NH-])OC=O', 'CC(=O)O[CH+][NH-]', '[NH-][CH+]OC(N)=O', 'C[NH+](C)CC#N', 'C[NH+](C)CC=O',
         '[NH3+]CCC(=O)[O-]', 'NC(=[NH2+])C(=O)[O-]', 'C[NH+]1CC1(C)C', 'CC1CC([NH-])O1', 'C[NH+]1CC(=O)C1',
         'C[NH+]1CC(O)C1', 'CC12C[NH+](C1)C2', 'OC12C[NH+](C1)C2', 'C/N=C(\\N)C#N', 'C/N=C(\\N)C=O', 'C[NH2+]C(C)OC']
        for s in smiles:
            print("Featurizing smile 1: " + s)
            g = featurizer.featurize(s)
            g_def = featurizer.defeaturize(g)
            smile_2 = Chem.MolToSmiles(g_def[0])
            print("Deafeturized smile 1 : " + smile_2)

            smile_3 = "C[NH+]1CC(O)C1"
            g_2 = featurizer.featurize(smile_3)
            print("COMPARING GRAPHS")
            self.assertEqual(g, g)

    def test_dataset(self, max_atoms=40, length=100):
        featurizer = MolGanFeaturizer(max_atom_count=max_atoms, kekulize=True, sanitize=True)
        self.dataset = SmilesDataset(smiles=self.smiles[0:length], task="generation", featurizer=featurizer)
        self.dataset.create()
        # for s in self.smiles:
        #     from jaqpotpy.cfg import config
        #     config.verbose = False
        #     print("------")
        #     mol = Chem.MolFromSmiles(s)
        #     # mol = Chem.Kekulize(mol)
        #     print(s)
        #     item = featurizer.featurize(s)
        #     mol = featurizer.defeaturize(item)
        #     try:
        #         smiles = Chem.MolToSmiles(mol[0])
        #     except Exception as e:
        #         continue
        #         print(e)
        # item = featurizer.featurize(self.smiles[0])
        # print(item[0].node_features.shape)
        # print(item[0].node_features)
        # print(item[0].adjacency_matrix.shape)
        # print(item[0].adjacency_matrix)
        # mol = featurizer.defeaturize(item)
        # print(Chem.MolToSmiles(mol[0]))
        # for i in range(len(self.dataset.smiles)):
        #
        #     ditem = self.dataset.__getitem__(i)
        #     print(self.dataset.smiles[i])
        #     g1 = GraphMatrix(ditem[2][1].cpu().detach().numpy(), ditem[2][0].cpu().detach().numpy())
        #     t = featurizer.defeaturize(g1)
        #     print(Chem.MolToSmiles(t[0]))
        #
        #
        # data_loader = DataLoader(dataset=self.dataset, **{'batch_size': 4, 'shuffle': False, 'num_workers': 0})
        #
        # items = self.dataset.smiles
        # print(items)
        # print(len(self.dataset.smiles))
        # for ditems in data_loader:
        #     nod_fs = torch.split(ditems[2][0], 1)
        #     adc_mat = torch.split(ditems[2][1], 1)
        #     for ind, n in enumerate(nod_fs):
        #         g2 = GraphMatrix(torch.squeeze(adc_mat[ind]).cpu().detach().numpy(), torch.squeeze(n).cpu().detach().numpy())
        #         t = featurizer.defeaturize(g2)
                # print(Chem.MolToSmiles(t[0]))

    def test_defeat(self):
        # smiles = "N.[CH2-]c1cc(O)co1"
        smiles = "C=C1C=CC(=O)O1.N"
        mol = Chem.MolFromSmiles(smiles)
        print(smiles)
        feat = MolGanFeaturizer(max_atom_count=42, kekulize=True, sanitize=True)
        f = feat.featurize(smiles)
        print(f)
        mol = feat.defeaturize(f)
        print(Chem.MolToSmiles(mol))

    def test_defeat_two(self):
        print(self.smiles[1])
        mol = Chem.MolFromSmiles(self.smiles[1])
        feat = MolGanFeaturizer(max_atom_count=42)
        f = feat.featurize(mol)
        print(f)
        mol = feat.defeaturize(f)
        print(Chem.MolToSmiles(mol))

    def test_generator_model(self):
        max_atoms = 40
        self.test_dataset(max_atoms=max_atoms)
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, max_atoms, 0.5)
        print(generator.eval())
        data_loader = DataLoader(dataset=self.dataset, **{'batch_size': 1, 'shuffle': False, 'num_workers': 0})
        for i in data_loader:
            # print(i)
            samples = generator.sample_generator(data_loader.batch_size)
            # print(samples)

            out1, out2 = generator(samples)
            g = GraphMatrix(out1.cpu().detach().numpy(), out2.cpu().detach().numpy())
            self.dataset.featurizer.defeaturize(g)
            print(out1.size())
            print(out2.size())
            edges_hat, nodes_hat = self.postprocess((out1, out2), 'soft_gumbel')
            # print(edges_hat.size())
            # print(edges_hat)
            # print(nodes_hat.size())

    def test_discriminator(self):
        import torch
        max_atoms = 42
        atoms_encoded = 12
        self.test_dataset(max_atoms=max_atoms)
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, atoms_encoded, 0.5)
        discriminator = MoleculeDiscriminator([[128, 64], 128, [128, 64]], atoms_encoded, 5 - 1, 0.5)
        data_loader = DataLoader(dataset=self.dataset, **{'batch_size': 4, 'shuffle': False, 'num_workers': 0})
        for i in data_loader:
            j = i[0]
            # print(i[0])
            samples = generator.sample_generator(data_loader.batch_size)
            # print(samples)
            out1, out2 = generator(samples)

            print(out1.size())
            print(out2.size())

            edges_hard, nodes_hard = torch.max(out1, -1)[1], torch.max(out2, -1)[1]


            out_d = discriminator(out1, None, out2)
            print(out_d)
            p = self.postprocess((out1, out2), "hard_gumbel")

            print(p[0].shape)
            print(p[1].shape)
            g = GraphMatrix(torch.squeeze(p[0]).cpu().detach().numpy(), torch.squeeze(p[1]).cpu().detach().numpy())
            mol = self.dataset.featurizer.defeaturize(g)
            print(mol)

            nod_fs = torch.split(i[2][0], 1)
            adc_mat = torch.split(i[2][1], 1)
            for ind, n in enumerate(nod_fs):
                g2 = GraphMatrix(torch.squeeze(adc_mat[ind]).cpu().detach().numpy(), torch.squeeze(n).cpu().detach().numpy())
                t = self.dataset.featurizer.defeaturize(g2)
                print(Chem.MolToSmiles(t[0]))

    def test_generator_optim(self):
        import numpy as np
        max_atoms = 42
        atoms_encoded = 12
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, atoms_encoded, 0.5)
        g_optim = torch.optim.Adam(generator.parameters(), lr=0.1, weight_decay=0.01)
        loss = torch.from_numpy(np.full(3, 0.3))
        loss = torch.mean(loss)
        loss.requires_grad = True
        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

    def test_gan_solver(self):
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.info')

        from jaqpotpy.cfg import config
        smiles_len = 1000
        config.verbose = False
        max_atoms = 42
        atoms_encoded = 12
        self.test_dataset(max_atoms=max_atoms, length=smiles_len)
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, atoms_encoded, 0.5)
        discriminator = MoleculeDiscriminator([[128, 64], 128, [128, 64]], atoms_encoded, 5 - 1, 0.5)
        solver = GanSolver(generator=generator
                           , discriminator=discriminator, dataset=self.dataset
                           , la=0.1, g_lr=0.1, d_lr=0.1, batch_size=42, val_at=20, epochs=30)
        solver.fit()

    def test_with_qm_9(self):
        mols = Chem.SDMolSupplier('./data/gdb9.sdf')
        max_atoms = 10
        atoms_encoded = 6
        smiles_gen = []
        for m in mols:
            try:
                smile = Chem.MolToSmiles(m)
                smiles_gen.append(smile)
            except Exception as e:
                continue
        featurizer = MolGanFeaturizer(max_atom_count=10, kekulize=True, sanitize=True, atom_labels=[0, 6, 7, 8, 9, 5])
        self.dataset = SmilesDataset(smiles=smiles_gen[1000:5000], task="generation", featurizer=featurizer)
        self.dataset.create()
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, atoms_encoded, 0.5)
        discriminator = MoleculeDiscriminator([[128, 64], 128, [128, 64]], atoms_encoded, 5 - 1, 0.5)
        solver = GanSolver(generator=generator
                           , discriminator=discriminator, dataset=self.dataset
                           , la=1, g_lr=0.1, d_lr=0.1, batch_size=42, val_at=20, epochs=30)
        solver.fit()
