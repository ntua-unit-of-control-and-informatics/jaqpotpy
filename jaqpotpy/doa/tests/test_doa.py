"""Tests for doa Methods."""

import unittest
import numpy as np
from jaqpotpy.descriptors.molecular import RDKitDescriptors
from jaqpotpy.doa import (
    Leverage,
    MeanVar,
    BoundingBox,
    Mahalanobis,
    KernelBased,
    CityBlock,
)


class TestDoa(unittest.TestCase):
    def test_leverage(self):
        mols = [
            "O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1",
            "O=C1CCc2cc(C(=O)N3CCC(C4CCNC4)CC3)ccc2N1",
            "CCC(=O)Nc1ccc(N(Cc2ccccc2)C(=O)n2nnc3ccccc32)cc1",
            "COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1",
            "Cc1nn(C)c2[nH]nc(NC(=O)Cc3cccc(Cl)c3)c12",
            "O=C(Cc1cncc2ccccc12)N(CCC1CCCCC1)c1cccc(Cl)c1",
            "COc1ccc(N(Cc2ccccc2)C(=O)Cc2c[nH]c3ccccc23)cc1",
            "CC(C)(C)c1ccc(N(C(=O)c2ccco2)[C@H](C(=O)NCCc2cccc(F)c2)c2cccnc2)cc1",
            "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2",
            "Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2cnn(C)c2C(F)F)c1",
            "Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)Cc1cccc(Cl)c1",
            "Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)C1CCOc2ccc(Cl)cc21",
            "O=C(c1cc(=O)[nH]c2ccccc12)N1CCN(c2cccc(Cl)c2)C(=O)C1",
            "O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12",
            "COCCNC(=O)[C@@H](c1ccccc1)N1Cc2ccccc2C1=O",
            "CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1",
            "O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12",
            "COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1",
            "O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21",
            "COc1ccccc1OC1CCN(C(=O)c2cc(=O)[nH]c3ccccc23)C1",
            "O=C(Cc1cc(Cl)cc(Cc2ccn[nH]2)c1)Nc1cncc2ccccc12",
            "CN(C)c1ccc(N(Cc2ccsc2)C(=O)Cc2cncc3ccccc23)cc1",
            "C[C@H]1COc2ccc(Cl)cc2[C@@H]1C(=O)Nc1cncc2ccccc12",
        ]

        featurizer = RDKitDescriptors(use_fragment=False, ipc_avg=False)
        descriptors = featurizer(mols)

        doa = Leverage()
        doa.fit(descriptors)
        mol = [
            "C[C@@](C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO",
            "COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1",
            "CN(C)c1ccc(N(Cc2ccsc2)C(=O)Cc2cncc3ccccc23)cc1",
        ]

        descriptors = featurizer(mol)
        calc = doa.predict(descriptors)

        assert len(calc) == len(
            mol
        ), f"Expected len(calc) == len(mol), got {len(calc)} != {len(mol)}"
        assert (
            abs(doa.h_star - 16.434782608695652) < 0.00001
        ), f"Expected doa.a == 16.434782608695652, got {doa.h_star} != 16.434782608695652"
        assert not calc[0][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == False, got {calc[0]['inDoa']} != False"
        assert calc[1][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == True, got {calc[1]['inDoa']} != True"
        assert calc[2][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == True, got {calc[2]['inDoa']} != True"

    def test_MeanVar(self):
        mols = [
            "C[C@@](C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO",
            "O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1",
            "O=C1CCc2cc(C(=O)N3CCC(C4CCNC4)CC3)ccc2N1",
            "CCC(=O)Nc1ccc(N(Cc2ccccc2)C(=O)n2nnc3ccccc32)cc1",
            "COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1",
            "Cc1nn(C)c2[nH]nc(NC(=O)Cc3cccc(Cl)c3)c12",
            "O=C(Cc1cncc2ccccc12)N(CCC1CCCCC1)c1cccc(Cl)c1",
            "COc1ccc(N(Cc2ccccc2)C(=O)Cc2c[nH]c3ccccc23)cc1",
            "CC(C)(C)c1ccc(N(C(=O)c2ccco2)[C@H](C(=O)NCCc2cccc(F)c2)c2cccnc2)cc1",
            "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2",
            "Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2cnn(C)c2C(F)F)c1",
            "Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)Cc1cccc(Cl)c1",
            "Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)C1CCOc2ccc(Cl)cc21",
            "O=C(c1cc(=O)[nH]c2ccccc12)N1CCN(c2cccc(Cl)c2)C(=O)C1",
            "O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12",
        ]

        featurizer = RDKitDescriptors(use_fragment=False, ipc_avg=False)
        descriptors = featurizer(mols)

        doa = MeanVar()
        doa.fit(descriptors)

        mol = [
            "C[C@@](C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO",
            "CCC",
        ]
        descriptors = featurizer(mol)
        calc = doa.predict(descriptors)
        diag = np.diag(doa.bounds)

        assert len(calc) == len(mol)
        assert not calc[0][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == False, got {calc[0]['inDoa']} != False"
        assert not calc[1][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == False, got {calc[1]['inDoa']} != False"
        assert np.allclose(
            diag, [1.31511044e01, 6.69162726e-01, 5.37187947e-03], atol=1e-5
        ), f"Expected diag == [1.31511044e+01, 6.69162726e-01, 5.37187947e-03], got diag != {diag}"

    def test_BoundingBox(self):
        mols = [
            "C[C@@](C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO",
            "O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1",
            "O=C1CCc2cc(C(=O)N3CCC(C4CCNC4)CC3)ccc2N1",
            "CCC(=O)Nc1ccc(N(Cc2ccccc2)C(=O)n2nnc3ccccc32)cc1",
            "COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1",
            "Cc1nn(C)c2[nH]nc(NC(=O)Cc3cccc(Cl)c3)c12",
            "O=C(Cc1cncc2ccccc12)N(CCC1CCCCC1)c1cccc(Cl)c1",
            "COc1ccc(N(Cc2ccccc2)C(=O)Cc2c[nH]c3ccccc23)cc1",
            "CC(C)(C)c1ccc(N(C(=O)c2ccco2)[C@H](C(=O)NCCc2cccc(F)c2)c2cccnc2)cc1",
            "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2",
            "Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2cnn(C)c2C(F)F)c1",
            "Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)Cc1cccc(Cl)c1",
            "Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)C1CCOc2ccc(Cl)cc21",
            "O=C(c1cc(=O)[nH]c2ccccc12)N1CCN(c2cccc(Cl)c2)C(=O)C1",
            "O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12",
        ]

        featurizer = RDKitDescriptors(use_fragment=False, ipc_avg=False)
        descriptors = featurizer(mols)
        print(type(descriptors))
        doa = BoundingBox()
        doa.fit(descriptors)

        mol = [
            "C[C@@](C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO",
            "CCC",
        ]
        descriptors = featurizer(mol)
        calc = doa.predict(descriptors)
        first_feature_bounds = doa.bounding_box[0]
        last_feature_bounds = doa.bounding_box[-1]
        print("Bounding box")
        assert len(calc) == len(mol)
        assert calc[0][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == True, got {calc[0]['inDoa']} != True"
        assert not calc[1][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == False, got {calc[1]['inDoa']} != False"
        assert np.allclose(
            first_feature_bounds, [12.0648171, 14.92728396], atol=1e-5
        ), f"Expected first_feature_bounds == [12.0648171 , 14.92728396], got {first_feature_bounds} != [12.0648171 , 14.92728396]"
        assert np.allclose(
            last_feature_bounds, [7.22405000e01, 2.40083000e02], atol=1e-5
        ), f"Expected last_feature_bounds == [7.22405000e+01,  2.40083000e+02], got {last_feature_bounds} !=  [7.22405000e+01,  2.40083000e+02]"

    def test_mahalanobis(self):
        mols = [
            "O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1",
            "O=C1CCc2cc(C(=O)N3CCC(C4CCNC4)CC3)ccc2N1",
            "CCC(=O)Nc1ccc(N(Cc2ccccc2)C(=O)n2nnc3ccccc32)cc1",
            "COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1",
            "Cc1nn(C)c2[nH]nc(NC(=O)Cc3cccc(Cl)c3)c12",
            "O=C(Cc1cncc2ccccc12)N(CCC1CCCCC1)c1cccc(Cl)c1",
            "COc1ccc(N(Cc2ccccc2)C(=O)Cc2c[nH]c3ccccc23)cc1",
            "CC(C)(C)c1ccc(N(C(=O)c2ccco2)[C@H](C(=O)NCCc2cccc(F)c2)c2cccnc2)cc1",
            "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2",
            "Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2cnn(C)c2C(F)F)c1",
            "Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)Cc1cccc(Cl)c1",
            "Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)C1CCOc2ccc(Cl)cc21",
            "O=C(c1cc(=O)[nH]c2ccccc12)N1CCN(c2cccc(Cl)c2)C(=O)C1",
            "O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12",
            "COCCNC(=O)[C@@H](c1ccccc1)N1Cc2ccccc2C1=O",
            "CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1",
            "O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12",
            "COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1",
            "O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21",
            "COc1ccccc1OC1CCN(C(=O)c2cc(=O)[nH]c3ccccc23)C1",
            "O=C(Cc1cc(Cl)cc(Cc2ccn[nH]2)c1)Nc1cncc2ccccc12",
            "CN(C)c1ccc(N(Cc2ccsc2)C(=O)Cc2cncc3ccccc23)cc1",
            "C[C@H]1COc2ccc(Cl)cc2[C@@H]1C(=O)Nc1cncc2ccccc12",
        ]

        featurizer = RDKitDescriptors(use_fragment=False, ipc_avg=False)
        descriptors = featurizer(mols)

        doa = Mahalanobis()
        doa.fit(descriptors)
        mol = [
            "C[C@@](C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO",
            "COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1",
            "CN(C)c1ccc(N(Cc2ccsc2)C(=O)Cc2cncc3ccccc23)cc1",
        ]

        descriptors = featurizer(mol)
        calc = doa.predict(descriptors)

        assert len(calc) == len(
            mol
        ), f"Expected len(calc) == len(mol), got {len(calc)} != {len(mol)}"
        assert (
            abs(doa._threshold - 12.287775633348163) < 0.00001
        ), f"Expected doa.a == 12.287775633348163, got {doa._threshold} != 12.287775633348163"
        assert not calc[0][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == False, got {calc[0]['inDoa']} != False"
        assert calc[1][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == True, got {calc[1]['inDoa']} != True"
        assert calc[2][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == True, got {calc[2]['inDoa']} != True"

    def test_kernelbased(self):
        mols = [
            "O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1",
            "O=C1CCc2cc(C(=O)N3CCC(C4CCNC4)CC3)ccc2N1",
            "CCC(=O)Nc1ccc(N(Cc2ccccc2)C(=O)n2nnc3ccccc32)cc1",
            "COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1",
            "Cc1nn(C)c2[nH]nc(NC(=O)Cc3cccc(Cl)c3)c12",
            "O=C(Cc1cncc2ccccc12)N(CCC1CCCCC1)c1cccc(Cl)c1",
            "COc1ccc(N(Cc2ccccc2)C(=O)Cc2c[nH]c3ccccc23)cc1",
            "CC(C)(C)c1ccc(N(C(=O)c2ccco2)[C@H](C(=O)NCCc2cccc(F)c2)c2cccnc2)cc1",
            "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2",
            "Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2cnn(C)c2C(F)F)c1",
            "Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)Cc1cccc(Cl)c1",
            "Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)C1CCOc2ccc(Cl)cc21",
            "O=C(c1cc(=O)[nH]c2ccccc12)N1CCN(c2cccc(Cl)c2)C(=O)C1",
            "O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12",
            "COCCNC(=O)[C@@H](c1ccccc1)N1Cc2ccccc2C1=O",
            "CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1",
            "O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12",
            "COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1",
            "O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21",
            "COc1ccccc1OC1CCN(C(=O)c2cc(=O)[nH]c3ccccc23)C1",
            "O=C(Cc1cc(Cl)cc(Cc2ccn[nH]2)c1)Nc1cncc2ccccc12",
            "CN(C)c1ccc(N(Cc2ccsc2)C(=O)Cc2cncc3ccccc23)cc1",
            "C[C@H]1COc2ccc(Cl)cc2[C@@H]1C(=O)Nc1cncc2ccccc12",
        ]

        featurizer = RDKitDescriptors(use_fragment=False, ipc_avg=False)
        descriptors = featurizer(mols)

        doa = KernelBased()
        doa.fit(descriptors)
        mol = [
            "C[C@@](C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO",
            "COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1",
            "CN(C)c1ccc(N(Cc2ccsc2)C(=O)Cc2cncc3ccccc23)cc1",
        ]

        descriptors = featurizer(mol)
        calc = doa.predict(descriptors)

        assert len(calc) == len(
            mol
        ), f"Expected len(calc) == len(mol), got {len(calc)} != {len(mol)}"
        assert (
            abs(doa._threshold - 0.0) < 0.00001
        ), f"Expected doa.a ==  0.0, got {doa._threshold} !=  0.0"
        assert calc[0][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == True, got {calc[0]['inDoa']} != True"
        assert calc[1][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == True, got {calc[1]['inDoa']} != True"
        assert calc[2][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == True, got {calc[2]['inDoa']} != True"

    def test_cityblock(self):
        mols = [
            "O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1",
            "O=C1CCc2cc(C(=O)N3CCC(C4CCNC4)CC3)ccc2N1",
            "CCC(=O)Nc1ccc(N(Cc2ccccc2)C(=O)n2nnc3ccccc32)cc1",
            "COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1",
            "Cc1nn(C)c2[nH]nc(NC(=O)Cc3cccc(Cl)c3)c12",
            "O=C(Cc1cncc2ccccc12)N(CCC1CCCCC1)c1cccc(Cl)c1",
            "COc1ccc(N(Cc2ccccc2)C(=O)Cc2c[nH]c3ccccc23)cc1",
            "CC(C)(C)c1ccc(N(C(=O)c2ccco2)[C@H](C(=O)NCCc2cccc(F)c2)c2cccnc2)cc1",
            "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2",
            "Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2cnn(C)c2C(F)F)c1",
            "Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)Cc1cccc(Cl)c1",
            "Cc1cc(C(F)(F)F)nc2c1c(N)nn2C(=O)C1CCOc2ccc(Cl)cc21",
            "O=C(c1cc(=O)[nH]c2ccccc12)N1CCN(c2cccc(Cl)c2)C(=O)C1",
            "O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12",
            "COCCNC(=O)[C@@H](c1ccccc1)N1Cc2ccccc2C1=O",
            "CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1",
            "O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12",
            "COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1",
            "O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21",
            "COc1ccccc1OC1CCN(C(=O)c2cc(=O)[nH]c3ccccc23)C1",
            "O=C(Cc1cc(Cl)cc(Cc2ccn[nH]2)c1)Nc1cncc2ccccc12",
            "CN(C)c1ccc(N(Cc2ccsc2)C(=O)Cc2cncc3ccccc23)cc1",
            "C[C@H]1COc2ccc(Cl)cc2[C@@H]1C(=O)Nc1cncc2ccccc12",
        ]

        featurizer = RDKitDescriptors(use_fragment=False, ipc_avg=False)
        descriptors = featurizer(mols)

        doa = CityBlock()
        doa.fit(descriptors)
        mol = [
            "C[C@@](C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO",
            "COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1",
            "CN(C)c1ccc(N(Cc2ccsc2)C(=O)Cc2cncc3ccccc23)cc1",
        ]

        descriptors = featurizer(mol)
        calc = doa.predict(descriptors)

        assert len(calc) == len(
            mol
        ), f"Expected len(calc) == len(mol), got {len(calc)} != {len(mol)}"
        assert (
            abs(doa._threshold - 11811319.56736) < 0.00001
        ), f"Expected doa.a == 11811319.56736, got {doa._threshold} != 11811319.56736"
        assert not calc[0][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == False, got {calc[0]['inDoa']} != False"
        assert calc[1][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == True, got {calc[1]['inDoa']} != True"
        assert calc[2][
            "inDoa"
        ], f"Expected calc[0]['inDoa'] == True, got {calc[2]['inDoa']} != True"
