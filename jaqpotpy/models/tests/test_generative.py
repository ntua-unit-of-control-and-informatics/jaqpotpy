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

    def test_dataset(self, max_atoms=40):
        featurizer = MolGanFeaturizer(max_atom_count=max_atoms)
        self.dataset = SmilesDataset(smiles=self.smiles[0:10], task="generation", featurizer=featurizer)
        self.dataset.create()

        item = featurizer.featurize(self.smiles[0])
        # print(item[0].node_features.shape)
        # print(item[0].node_features)
        # print(item[0].adjacency_matrix.shape)
        # print(item[0].adjacency_matrix)
        mol = featurizer.defeaturize(item)
        # print(Chem.MolToSmiles(mol[0]))

        ditem = self.dataset.__getitem__(0)
        # print(ditem[2])
        print(ditem[2][0].numpy().shape)
        print(ditem[2][1].numpy().shape)
        g1 = GraphMatrix(ditem[2][1].cpu().detach().numpy(), ditem[2][0].cpu().detach().numpy())
        t = featurizer.defeaturize(g1)
        print(Chem.MolToSmiles(t[0]))
        data_loader = DataLoader(dataset=self.dataset, **{'batch_size': 4, 'shuffle': False, 'num_workers': 0})
        for ditems in data_loader:
            nod_fs = torch.split(ditems[2][0], 1)
            adc_mat = torch.split(ditems[2][1], 1)
            for ind, n in enumerate(nod_fs):
                g2 = GraphMatrix(torch.squeeze(adc_mat[ind]).cpu().detach().numpy(), torch.squeeze(n).cpu().detach().numpy())
                t = featurizer.defeaturize(g2)
                print(Chem.MolToSmiles(t[0]))

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
        max_atoms = 32
        atoms_encoded = 12
        self.test_dataset(max_atoms=max_atoms)
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, atoms_encoded, 0.5)
        t = generator.sample(20)
        discriminator = MoleculeDiscriminator([[128, 64], 128, [128, 64]], atoms_encoded, 5 - 1, 0.5)
        # print(generator.eval())
        # print(discriminator.eval())
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


    def test_gan_solver(self):
        max_atoms = 32
        atoms_encoded = 12
        self.test_dataset(max_atoms=max_atoms)
        generator = GanMoleculeGenerator([128, 256, 524], 8, max_atoms, 5, atoms_encoded, 0.5)
        t = generator.sample(20)
        discriminator = MoleculeDiscriminator([[128, 64], 128, [128, 64]], atoms_encoded, 5 - 1, 0.5)
        solver = GanSolver()


    def postprocess(self, inputs, method, temperature=1.):

        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                       / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

