"""
Tests for doa Methods.
"""
import unittest
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from jaqpotpy.descriptors.molecular import RDKitDescriptors, MordredDescriptors
from jaqpotpy.doa.doa import Leverage, MeanVar, SmilesLeverage


class TestDoa(unittest.TestCase):

    def test_leverage(self):
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

        featurizer = RDKitDescriptors(use_fragment=False, ipc_avg=False)
        descriptors = featurizer(mols)

        doa = Leverage()
        doa.fit(descriptors)
        mol = [
            'C[C@@](C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO',
            'COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1', 'CN(C)c1ccc(N(Cc2ccsc2)C(=O)Cc2cncc3ccccc23)cc1']
        
        descriptors = featurizer(mol)
        calc = doa.predict(descriptors)

        assert len(calc)==len(mol), f"Expected len(calc) == len(mol), got {len(calc)} != {len(mol)}"
        assert abs(doa.a - 16.434782608695652) < 0.00001, f"Expected doa.a == 16.434782608695652, got {doa.a} != 16.434782608695652"
        assert calc[0]['IN']==False, f"Expected calc[0]['IN'] == False, got {calc[0]['IN']} != False"
        assert calc[1]['IN']==True, f"Expected calc[0]['IN'] == True, got {calc[1]['IN']} != True"
        assert calc[2]['IN']==True, f"Expected calc[0]['IN'] == True, got {calc[2]['IN']} != True"
        

    @unittest.skip("This test needs refactoring")
    def test_smiles_leverage(self):
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

        doa = SmilesLeverage()
        doa.fit(mols)
        mol = [
            'C[C@@](C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO',
            'COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
            , 'CN(C)c1ccc(N(Cc2ccsc2)C(=O)Cc2cncc3ccccc23)cc1'
        ]
        calc = doa.predict(mol)
        assert doa.IN == [False, True, True]
        assert doa.doa_new == [90575896122526.53, 0.9804306739393107, 0.9992936436413169]
        assert len(calc) == len(mol)

    @unittest.skip("This test needs refactoring")
    def test_mean_var(self):
        mols = [
            'C[C@@](C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO',
            'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
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
        ]

        featurizer = RDKitDescriptors(use_fragment=False, ipc_avg=False)
        # featurizer = MordredDescriptors()

        descriptors = featurizer(mols)
        # descriptors = np.array([[0,1,2], [1,2,3], [1,2,1], [2,2,2], [3,3,3], [1,1,1], [2,1,3], [2,2,2], [2,2,1]])

        minmax = MinMaxScaler()
        # doa = MeanVar(minmax)
        doa = MeanVar()

        doa.fit(descriptors)
        # doa.data = descriptors
        # doa.calculate_matrix()
        # doa.calculate_threshold()

        mol = [
            'C[C@@](C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO',
            'COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
            , 'O=C(Cc1cncc2ccccc12)N(CCC1CCCCC1)c1cccc(Cl)c1'
            , 'Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2cnn(C)c2C(F)F)c1'
            , 'OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2'
            , 'Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2cnn(C)c2C(F)F)c1'
            , 'Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)Cc1cccc(Cl)c1'
            , 'Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)C1CCOc2ccc(Cl)cc21'
            , 'O=C(c1cc(=O)[nH]c2ccccc12)N1CCN(c2cccc(Cl)c2)C(=O)C1'
            , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
        ]
        descriptors = featurizer(mol)
        # descriptors = np.array([[5,1,0], [3,3,3], [-1,-2,0], [5,5,5], [100,10,40]])
        calc = doa.predict(descriptors)
        # print(doa.a)
        # print(doa.doa_matrix)
        # print(calc)
        assert len(calc) == len(mol)

    @unittest.skip("This test needs refactoring")
    def test_with_other_data(self):
        # basedir = os.path.dirname(sys.argv[0])
        # filename = "gdp-countries.csv"
        # path = os.path.join(basedir, "results", filename)

        # data = pd.read_csv(path)
        # data = pd.read_csv('../../test_data/gdp-countries.csv')

        # data = data[['GDP', 'LFG', 'EQP', 'NEQ', 'GAP']].to_numpy()

        data = np.array([[0.0089, 0.0118, 0.0214, 0.2286, 0.6079], [0.0332, 0.0014, 0.0991, 0.1349, 0.5809],
                         [0.0256, 0.0061, 0.0684, 0.1653, 0.4109], [0.0124, 0.0209, 0.0167, 0.1133, 0.8634],
                         [0.0676, 0.0239, 0.131, 0.149, 0.9474], [0.0437, 0.0306, 0.0646, 0.1588, 0.8498]])

        minmax = MinMaxScaler()
        # doa = Leverage(minmax)
        doa = Leverage()
        doa.fit(data)
        calc = doa.predict(data)
        assert len(calc) == len(data)

    @unittest.skip("This test needs refactoring")
    def test_with_other_data_mean_var(self):
        # basedir = os.path.dirname(sys.argv[0])
        # filename = "gdp-countries.csv"
        # path = os.path.join(basedir, "results", filename)
        #
        # data = pd.read_csv(path)

        # data = pd.read_csv('../../test_data/gdp-countries.csv')
        # data = data[['GDP', 'LFG', 'EQP', 'NEQ', 'GAP']].to_numpy()

        data = np.array([[0.0089, 0.0118, 0.0214, 0.2286, 0.6079], [0.0332, 0.0014, 0.0991, 0.1349, 0.5809],
                         [0.0256, 0.0061, 0.0684, 0.1653, 0.4109], [0.0124, 0.0209, 0.0167, 0.1133, 0.8634],
                         [0.0676, 0.0239, 0.131, 0.149, 0.9474], [0.0437, 0.0306, 0.0646, 0.1588, 0.8498]])

        minmax = MinMaxScaler()
        # doa = Leverage(minmax)
        doa = MeanVar()
        doa.fit(data)
        calc = doa.predict(data)
        assert len(calc) == len(data)