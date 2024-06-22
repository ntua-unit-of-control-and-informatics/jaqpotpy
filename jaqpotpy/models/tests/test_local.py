"""
Tests for Jaqpotpy Models.
"""
import unittest
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.descriptors.molecular import MordredDescriptors\
    , create_char_to_idx, SmilesToSeq, OneHotSequence, SmilesToImage\
    , TopologicalFingerprint, RDKitDescriptors, MACCSKeysFingerprint

from jaqpotpy.descriptors.molecular.molecule_graph_conv import MolGraphConvFeaturizer\
  , PagtnMolGraphFeaturizer, TorchMolGraphConvFeaturizer, AttentiveFPFeaturizer
from jaqpotpy.models import MolecularModel, MolecularSKLearn
from sklearn.linear_model import LinearRegression
import asyncio
from jaqpotpy.doa.doa import Leverage
from jaqpotpy.models.evaluator import Evaluator
from jaqpotpy.models.preprocessing import Preprocesses
from sklearn.metrics import max_error, mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch.autograd import Variable
from jaqpotpy.models import MolecularTorchGeometric, MolecularTorch
from jaqpotpy.models.torch_models import AttentiveFP_V1
import jaqpotpy.utils.pytorch_utils as ptu
from jaqpotpy.descriptors.molecular import MolGraphConvFeaturizer
from torch_geometric.loader import DataLoader
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch_geometric
from jaqpotpy.models.torch_models import GCN_V1 as GCN_J
from jaqpotpy import Jaqpot
from sklearn import svm
from rdkit import Chem
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
    mols = ['O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
        , 'O=C1CCc2cc(C(=O)N3CCC(C4CCNC4)CC3)ccc2N1'
        , 'CCC(=O)Nc1ccc(N(Cc2ccccc2)C(=O)n2nnc3ccccc32)cc1'
        , 'COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
        , 'Cc1nn(C)c2[nH]nc(NC(=O)Cc3cccc(Cl)c3)c12'
        , 'O=C(Cc1cncc2ccccc12)N(CCC1CCCCC1)c1cccc(Cl)c1'
        , 'COc1ccc(N(Cc2ccccc2)C(=O)Cc2c[nH]c3ccccc23)cc1'
        , 'CC(C)(C)c1ccc(N(C(=O)c2ccco2)[C@H](C(=O)NCCc2cccc(F)c2)c2cccnc2)cc1'
        , 'OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2'
        , 'Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2cnn(C)c2C(F)F)c1'
        , 'Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)Cc1cccc(Cl)c1'
        , 'Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)C1CCOc2ccc(Cl)cc21'
        , 'O=C(c1cc(=O)[nH]c2ccccc12)N1CCN(c2cccc(Cl)c2)C(=O)C1'
        , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
        , 'COCCNC(=O)[C@@H](c1ccccc1)N1Cc2ccccc2C1=O'
        , 'CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1'
        , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
        , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
        , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21'
        , 'COc1ccccc1OC1CCN(C(=O)c2cc(=O)[nH]c3ccccc23)C1'
        , 'O=C(Cc1cc(Cl)cc(Cc2ccn[nH]2)c1)Nc1cncc2ccccc12'
        , 'CN(C)c1ccc(N(Cc2ccsc2)C(=O)Cc2cncc3ccccc23)cc1'
        , 'C[C@H]1COc2ccc(Cl)cc2[C@@H]1C(=O)Nc1cncc2ccccc12'
    ]

    ys = [
        0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1
    ]

    ys_regr = [
        0.001, 1.286, 2.8756, 1.021, 1.265, 0.0012, 0.0028, 0.987, 2.567
        , 1.0002, 1.008, 1.1234, 0.25567, 0.5647, 0.99887, 1.9897, 1.989, 2.314, 0.112, 0.113, 0.54, 1.123, 1.0001
    ]

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

    @unittest.skip('Local data')
    def test_model(self):
        # featurizer = MordredDescriptors(ignore_3D=True)
        featurizer = RDKitDescriptors()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , x_cols=['molregno', 'organism']
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          ,
                                          X=['nBase', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A', 'LogEE_A',
                                             'VE1_A', 'VE2_A']
                                          )
        model = LinearRegression()
        molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        assert molecularModel_t1.doa.IN == [False]
        molecularModel_t1.prediction[0]
        # [[-3196989.37288455]]

    @unittest.skip('Local data')
    def test_model_no_doa(self):
        # featurizer = MordredDescriptors(ignore_3D=True)
        featurizer = RDKitDescriptors()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          # , x_cols=['molregno', 'organism']
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          ,
                                          # X=['nBase', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A', 'LogEE_A',
                                          #    'VE1_A', 'VE2_A']
                                          )
        model = LinearRegression()
        molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1.save()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        assert molecularModel_t1.doa is not None
        # assert molecularModel_t1.doa.IN == [False]
        # assert molecularModel_t1.doa.doa_new == [271083.32573286095]
        # assert int(molecularModel_t1.prediction[0]) == -2873819

    @unittest.skip('Local data')
    def test_model_rdkit_pickle_2(self):
        featurizer = RDKitDescriptors()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          )
        model = svm.SVC()
        molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1.save()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        molecularModel_t1.prediction

    @unittest.skip('Local data')
    def test_model_rdkit_unpickle(self):
        model = MolecularModel().load("./jaqpot_model.jmodel")
        model('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        print(model.prediction)

    @unittest.skip('Local data')
    def test_model_fingerprint_pickle(self):
        featurizer = TopologicalFingerprint()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          )
        model = svm.SVC()
        molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1.save()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        print(molecularModel_t1.prediction)
        # assert molecularModel_t1.doa is None

    # def test_load_f_m(self):
    #     model = MolecularModel().load("./jaqpot_model.jmodel")
    #     model('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
    #     model.prediction
    @unittest.skip('Local data')
    def test_model_fingerprint_pickle_2(self):
        featurizer = TopologicalFingerprint()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          )
        model = svm.SVC()
        molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1.save()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        molecularModel_t1.prediction
        # assert molecularModel_t1.doa is None

    @unittest.skip('Local data')
    def test_model_topological_fingerprint_pickle(self):
        featurizer = TopologicalFingerprint()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          )
        model = svm.SVC()
        molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1.save()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        assert molecularModel_t1.doa is not None

    @unittest.skip('Local data')
    def test_model_rdkit_pickle(self):
        featurizer = RDKitDescriptors()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          )
        model = svm.SVC()
        molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1.save()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')

    @unittest.skip('Local data')
    def test_model_top_pickle(self):
        featurizer = TopologicalFingerprint()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          )
        model = svm.SVC()
        molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1.save()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        assert molecularModel_t1.doa is not None

    @unittest.skip('Local data')
    def test_model_smiles_doa(self):
        featurizer = MordredDescriptors(ignore_3D=True)
        # featurizer = RDKitDescriptors()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , x_cols=['molregno', 'organism']
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          ,
                                          X=['nBase', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A', 'LogEE_A',
                                             'VE1_A', 'VE2_A']
                                          )

        model = LinearRegression()
        molecularModel_t2 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t2('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        assert molecularModel_t2.doa.IN == [True]
        assert int(molecularModel_t2.doa.doa_new[0]) == 0
        # [[-3196989.37288455]]

    @unittest.skip('Local data')
    def test_model_no_feats_with_external(self):
        # featurizer = MordredDescriptors(ignore_3D=True)
        featurizer = RDKitDescriptors()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , x_cols=['molregno']
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          )
        model = LinearRegression()
        molecularModel_t3 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        ext = {'molregno': [100]}
        molecularModel_t3('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1', ext)
        assert molecularModel_t3.external_feats == ['molregno']
        print(int(molecularModel_t3.prediction[0][0]))
        assert int(molecularModel_t3.prediction[0][0]) == -80054

    @unittest.skip('Local data')
    def test_model_no_cols(self):
        # featurizer = MordredDescriptors(ignore_3D=False)
        featurizer = RDKitDescriptors()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          )
        model = LinearRegression()
        molecularModel_t4 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t4('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        molecularModel_t4.doa.IN
        molecularModel_t4.doa.doa_new
        molecularModel_t4.doa.a
        molecularModel_t4.prediction
        # assert int(molecularModel_t4.prediction[0][0]) == 21211

    @unittest.skip('Local data')
    def test_model_no_cols_with_eval(self):
        # featurizer = MordredDescriptors(ignore_3D=False)
        featurizer = RDKitDescriptors()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          )
        # dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = LinearRegression()
        molecularModel_t5 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=val).fit()
        molecularModel_t5('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        print(molecularModel_t5.prediction)
        # assert int(molecularModel_t5.prediction[0][0]) == 21232

    @unittest.skip('Local data')
    def test_model_pre(self):
        featurizer = RDKitDescriptors()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          )
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        pre = Preprocesses()
        pre.register_preprocess_class("Standard Scaler", StandardScaler())
        pre.register_preprocess_class_y("Standard Scaler", StandardScaler())
        model = LinearRegression()
        molecularModel_t6 = MolecularSKLearn(dataset=dataset
                                             , doa=Leverage()
                                             , model=model
                                             , eval=val
                                             , preprocess=pre).fit()
        molecularModel_t6(['COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
                              , 'CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1'
                              , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
                              , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
                              , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21'])
        # print(molecularModel.doa.IN)
        # print(molecularModel.doa.doa_new)
        # print(molecularModel.doa.a)
        # print(molecularModel.prediction)
        print(molecularModel_t6.jaqpotpy_docker)
        print(molecularModel_t6.prediction)
        # assert int(molecularModel_t6.prediction[0][0]) == 1228766

    @unittest.skip('Local data')
    def test_model_save(self):
        featurizer = RDKitDescriptors()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          )
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        pre = Preprocesses()
        pre.register_preprocess_class("Standard Scaler", StandardScaler())
        model = LinearRegression()
        molecularModel_t7 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=val,
                                             preprocess=pre).fit()
        molecularModel_t7(['COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
                              , 'CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1'
                              , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
                              , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
                              , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21'])
        # print(molecularModel_t7.doa.IN)
        # print(molecularModel_t7.doa.doa_new)
        # print(molecularModel_t7.doa.a)
        print(molecularModel_t7.prediction)
        molecularModel_t7.model_name = "TestModel"
        molecularModel_t7.save()
        # assert int(molecularModel_t7.prediction[0][0]) == 1228766

    @unittest.skip("Needs refactoring")
    def test_load_model(self):
        try:
            molecularModel_t8 = MolecularModel.load('./TestModel.jmodel')
            molecularModel_t8(['COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
                                  , 'CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1'
                                  , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
                                  , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
                                  , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21'])
            print(molecularModel_t8.doa.IN)
            print(molecularModel_t8.doa.doa_new)
            print(molecularModel_t8.doa.a)
            print(molecularModel_t8.prediction)
            molecularModel_t8.model_name = "TestModel"
            molecularModel_t8.save()
            assert int(molecularModel_t8.prediction[0][0]) == 1228766
            smiles_new = ['COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
                , 'CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1'
                , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
                , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
                , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21']
            for s in smiles_new:
                molecularModel_t8(s)
                print(molecularModel_t8.doa.IN)
                print(molecularModel_t8.prediction)
        except FileNotFoundError:
            print("A File is missing in load model")

    # This test meant to run local and not centralized
    @unittest.skip('Local data')
    def test_model_pre_torch(self):
        featurizer = MordredDescriptors(ignore_3D=True)
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(path=path
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          , task='regression'
                                          )
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model_nn = Feedforward()
        optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.L1Loss()

        m = MolecularTorch(dataset=dataset
                           , model_nn=model_nn, eval=val
                           , train_batch=4, test_batch=4
                           , epochs=30, optimizer=optimizer, criterion=criterion).fit()
        m.eval()
        molMod = m.create_molecular_model()
        molMod.model_title = "test_regression"
        molMod.save()
        molMod.load("./test_regression.jmodel")
        molMod(['COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
                   , 'CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1'
                   , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
                   , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
                   , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21'])
        # print(molMod.Y)
        # print(molMod.prediction)
        smiles_new = ['COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
            , 'CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1'
            , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
            , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
            , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21'
            , 'CC(C)(C)c1ccc(N(C(=O)c2ccco2)[C@H](C(=O)NCCc2cccc(F)c2)c2cccnc2)cc1'
            , 'OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2'
            , 'Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2cnn(C)c2C(F)F)c1']

        for smile in smiles_new:
            molMod(smile)
            # print(molMod.prediction)

    @unittest.skip('Local data')
    def test_model_class_torch(self):
        # featurizer = MordredDescriptors(ignore_3D=True)
        featurizer = RDKitDescriptors()
        path = '../../test_data/data.csv'
        dataset = JaqpotpyDataset(smiles=self.mols, y=self.ys, featurizer=featurizer, task="classification")
        # dataset = MolecularTabularDataset(path=path
        #                                   , y_cols=['standard_value']
        #                                   , smiles_col='canonical_smiles'
        #                                   , featurizer=featurizer
        #                                   , task='regression'
        #                                   )
        dataset.create()
        # dataset.y = 'activity'
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model_nn = Feedforward_class()
        optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        m = MolecularTorch(dataset=dataset
                           , model_nn=model_nn, eval=val
                           , train_batch=4, test_batch=4
                           , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        m.eval()
        molMod = m.create_molecular_model()
        molMod.model_name = "test_regression"
        molMod.save()
        try:
            molMod.load("./test_regression.jmodel")
        except FileNotFoundError:
            print("A File is missing in load model")
        # print(molMod.library)
        # print(molMod.version)
        # print(molMod.jaqpotpy_version)
        molMod(['COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
                   , 'CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1'
                   , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
                   , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
                   , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21'])
        # print(molMod.Y)
        # print(molMod.prediction)
        smiles_new = ['COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
            , 'CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1'
            , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
            , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
            , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21'
            , 'CC(C)(C)c1ccc(N(C(=O)c2ccco2)[C@H](C(=O)NCCc2cccc(F)c2)c2cccnc2)cc1'
            , 'OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2'
            , 'Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2cnn(C)c2C(F)F)c1']

        for smile in smiles_new:
            molMod(smile)


class CNNNet_regression(torch.nn.Module):
    def __init__(self):
        super(CNNNet_regression, self).__init__()

        self.cnn_layers = torch.nn.Sequential(
            # Defining a 2D convolution layer
            torch.nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            torch.nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = torch.nn.Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class CNNNet_classification(torch.nn.Module):
    def __init__(self):
        super(CNNNet_classification, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 12 * 12, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 2)

    def forward(self, x):
        torch.manual_seed(151)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class LSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size)

        self.linear = torch.nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(30, 40)
        self.conv2 = GCNConv(40, 40)
        self.conv3 = GCNConv(40, 40)
        self.lin = Linear(40, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class GCN_REGR(torch.nn.Module):
    def __init__(self):
        super(GCN_REGR, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(30, 40)
        self.conv2 = GCNConv(40, 40)
        self.conv3 = GCNConv(40, 40)
        self.lin = Linear(40, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class Feedforward(torch.nn.Module):
    def __init__(self, input_size=1613, hidden_size=3528):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        # output = self.sigmoid(output)
        return output


class Feedforward_class(torch.nn.Module):
    def __init__(self, input_size=1613, hidden_size=3528):
        super(Feedforward_class, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 2)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        # output = self.sigmoid(output)
        return output
