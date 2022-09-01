import unittest
from jaqpotpy.datasets.molecular_datasets import SmilesDataset
from jaqpotpy.descriptors.molecular import MolGanFeaturizer
from jaqpotpy.models.generative.models import GanGenerator, GanDiscriminator
from torch.utils.data import DataLoader
import torch.nn.functional as F


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

    def test_dataset(self):
        featurizer = MolGanFeaturizer(max_atom_count=40)
        self.dataset = SmilesDataset(smiles=self.smiles[:2], task="generation", featurizer=featurizer)
        self.dataset.create()

        item = featurizer.featurize(self.smiles[0])
        print(item)
        item = featurizer.defeaturize(item)
        print(item)

        item = self.dataset.__getitem__(1)
        print(item)
        t = featurizer.defeaturize(item)
        print(t)
        # print(item)

    def test_generator_model(self):
        self.test_dataset()
        generator = GanGenerator([128, 256, 524], 8, 40, 1, 40, 0.5)
        print(generator.eval())
        data_loader = DataLoader(dataset=self.dataset, **{'batch_size': 1, 'shuffle': False, 'num_workers': 0})
        for i in data_loader:
            # print(i)
            samples = generator.sample_generator(data_loader.batch_size)
            # print(samples)
            out1, out2 = generator(samples)
            print(out1.size())
            print(out2.size())
            edges_hat, nodes_hat = self.postprocess((out1, out2), 'soft_gumbel')
            # print(edges_hat.size())
            # print(edges_hat)
            # print(nodes_hat.size())


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
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

