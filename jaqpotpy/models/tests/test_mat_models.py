import unittest
from jaqpotpy.datasets import CompositionDataset, StructureDataset
from jaqpotpy.descriptors.material import ElementNet, CrystalGraphCNN,SineCoulombMatrix
from jaqpotpy.models import MaterialSKLearn, MaterialTorch
from sklearn.linear_model import LinearRegression
import asyncio
from pymatgen.core.structure import Structure, Lattice
from jaqpotpy.models.evaluator import Evaluator
from jaqpotpy.models.preprocessing import Preprocesses
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch.autograd import Variable
from jaqpotpy.models import MolecularTorchGeometric, MolecularTorch
import jaqpotpy.utils.pytorch_utils as ptu
from jaqpotpy.descriptors import MolGraphConvFeaturizer
from torch_geometric.loader import DataLoader
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# import pytest
from jaqpotpy.doa.doa import Leverage
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
def sync(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(coro(*args, **kwargs))

    return wrapper


# def async_test(f):
#     def wrapper(*args, **kwargs):
#         coro = asyncio.coroutine(f)
#         future = coro(*args, **kwargs)
#         loop = asyncio.get_event_loop()
#         loop.run_until_complete(future)
#     return wrapper


class TestModels(unittest.TestCase):


    def setUp(self):
        self.compositions = ['FeO', 'Fe3O2', 'FeHeO', 'ArFeO', 'FeHO']

        self.lattices = [Lattice.cubic(item) for item in [4.2, 5.3, 6.7, 2.5, 1.9]]
        self.atoms = ['Cs', 'Cl', 'Ca', 'Fe', 'Ag', 'Cs']
        self.coords = [
            [[0, 0, 0], [0.5, 0.5, 0.5]],
            [[0, 0, 0], [0.6, 0.6, 0.6]],
            [[0, 0, 0], [0.7, 0.7, 0.7]],
            [[0, 0, 0], [0.8, 0.8, 0.8]],
            [[0, 0, 0], [0.9, 0.9, 0.9]]
        ]
        self.path = 'jaqpotpy/test_data/comp_dataset_data.csv'

        self.ys = [
            0, 1, 1, 1, 0
        ]

        self.ys_regr = [
            0.001, 1.286, 2.8756, 1.021, 1.265,
        ]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    def test_sklearn_comp(self):
        import warnings
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)


        featurizer = ElementNet()
        dataset = CompositionDataset(compositions=self.compositions, y=self.ys_regr, task='regression', featurizer=featurizer)

        model = LinearRegression()
        material_model = MaterialSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        material_model('FeO')
        material_model.prediction

    def test_sklearn_struct(self):
        import warnings
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)


        featurizer = SineCoulombMatrix()
        structs = [
            Structure(self.lattices[i], [self.atoms[i], self.atoms[i + 1]], self.coords[i])
            for i in range(5)
        ]
        dataset = StructureDataset(structures=structs, y=self.ys_regr, featurizer=featurizer)

        model = LinearRegression()
        material_model = MaterialSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()

        material_model(structs[0])
        material_model.prediction
        # print(material_model.prediction)

    # def test_torch(self):
    #     featurizer = CrystalGraphCNN()
    #     structs = [
    #         Structure(self.lattices[i], [self.atoms[i], self.atoms[i + 1]], self.coords[i])
    #         for i in range(5)
    #     ]
    #     dataset = StructureDataset(structures=structs, y=self.ys_regr, featurizer=featurizer)
    #     print(dataset.df)


if __name__ == '__main__':
    unittest.main()
