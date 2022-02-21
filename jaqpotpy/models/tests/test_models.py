"""
Tests for Jaqpotpy Models.
"""
import unittest
from jaqpotpy.datasets.dataset_base import MolecularTabularDataset
from jaqpotpy.descriptors import MordredDescriptors
from jaqpotpy.models.base_classes import InMemMolModel
from sklearn.linear_model import LinearRegression
import asyncio
# import pytest
from jaqpotpy.doa.doa import Leverage
import numpy as np


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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    @sync
    async def test_async(self):
        async def fn():
            print('hello')
            await asyncio.sleep(1)
            print('world')
        await fn()

    def test_model(self):
        featurizer = MordredDescriptors(ignore_3D=True)
        path = '../../test_data/data.csv'
        dataset = MolecularTabularDataset(path=path
                                          , x_cols=['molregno', 'organism']
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          ,
                                          X=['nBase', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A', 'LogEE_A',
                                             'VE1_A', 'VE2_A']
                                          )

        model = LinearRegression()
        molecularModel = InMemMolModel(dataset=dataset, doa=Leverage(), model=model, eval=None).__train__()
        print(molecularModel.__dict__)
        print(molecularModel('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'))
        # pred = jmodel.__predict__('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        # jmodel.
        # print(pred)

        # jaqpotmodel("smiles")

        # dataset.create()
        # assert dataset.featurizer_name == 'MordredDescriptors'
        # assert dataset.x_cols == ['molregno', 'organism']
        # assert dataset.y_cols == ['standard_value']
        # assert dataset.smiles_strings[0] == 'CO[C@@H]1[C@@H](O)[C@@H](O)[C@H](Oc2ccc3c(O)c(NC(=O)/C=C/c4ccccc4)c(=O)oc3c2C)OC1(C)C'
        # assert dataset.df.shape == (4, 1830)
