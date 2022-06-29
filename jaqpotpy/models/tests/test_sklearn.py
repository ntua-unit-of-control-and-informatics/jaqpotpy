"""
Tests for Jaqpotpy Models.
"""
import unittest
import asyncio
import warnings
from jaqpotpy.descriptors.molecular import TopologicalFingerprint, RDKitDescriptors, MordredDescriptors
from jaqpotpy.datasets import SmilesDataset, MolecularTabularDataset
from jaqpotpy.models import MolecularSKLearn
from jaqpotpy.doa.doa import Leverage
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=DeprecationWarning)


def sync(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(coro(*args, **kwargs))

    return wrapper

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

    def test_SVR(self):
        featurizer = RDKitDescriptors()
        dataset = SmilesDataset(smiles=self.mols, y=self.ys_regr, task='regression', featurizer=featurizer)
        model = SVR()
        molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        assert molecularModel_t1.doa.IN == [False]
        assert int(molecularModel_t1.doa.doa_new[0]) == 271083
        assert int(molecularModel_t1.prediction[0][0]) == -2873819

    def test_SVM(self):
        featurizer = MordredDescriptors(ignore_3D=True)
        path = './test_data/data.csv'
        dataset = MolecularTabularDataset(path=path
                                          , x_cols=['molregno', 'organism']
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          ,
                                          X=['nBase', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A',
                                             'LogEE_A',
                                             'VE1_A', 'VE2_A']
                                          )
        model = LinearRegression()
        molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        assert molecularModel_t1.doa.IN == [False]
        assert int(molecularModel_t1.doa.doa_new[0]) == 271083
        assert int(molecularModel_t1.prediction[0][0]) == -2873819