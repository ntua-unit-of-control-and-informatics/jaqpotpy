"""
Tests for Jaqpotpy Datasets.
"""
import unittest
from jaqpotpy.datasets import SmilesDataset, TorchGraphDataset, MolecularTabularDataset
from jaqpotpy.descriptors import MordredDescriptors\
    , TopologicalFingerprint, MolGraphConvFeaturizer\
    , RDKitDescriptors, SmilesToSeq, create_char_to_idx, SmilesToImage
from mordred import descriptors
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as dl


class TestDatasets(unittest.TestCase):

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

    def test_smiles_tab_data_with_x(self):
        # featurizer = TopologicalFingerprint()
        featurizer = MordredDescriptors(ignore_3D=False)
        path_b = '../../test_data/data_big.csv'
        path = '../../test_data/small.csv'
        dataset = MolecularTabularDataset(path=path
                                          , x_cols=['molregno', 'organism']
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          ,
                                          X=['nBase', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A', 'LogEE_A',
                                             'VE1_A', 'VE2_A']
                                          )

        dataset.create()
        assert dataset.featurizer_name == 'MordredDescriptors'
        assert dataset.x_cols == ['molregno', 'organism']
        assert dataset.y_cols == ['standard_value']
        assert dataset.smiles_strings[0] == 'CO[C@@H]1[C@@H](O)[C@@H](O)[C@H](Oc2ccc3c(O)c(NC(=O)/C=C/c4ccccc4)c(=O)oc3c2C)OC1(C)C'
        assert dataset.df.shape == (4, 1829)

    def test_smiles_tab_data_without_x(self):
        featurizer = MordredDescriptors(ignore_3D=False)
        path = '../../test_data/small.csv'
        dataset = MolecularTabularDataset(path=path
                                          , y_cols=['standard_value']
                                          , smiles_col='canonical_smiles'
                                          , featurizer=featurizer
                                          )

        dataset.create()
        assert dataset.featurizer_name == 'MordredDescriptors'
        assert dataset.y_cols == ['standard_value']
        assert dataset.smiles_strings[
                   0] == 'CO[C@@H]1[C@@H](O)[C@@H](O)[C@H](Oc2ccc3c(O)c(NC(=O)/C=C/c4ccccc4)c(=O)oc3c2C)OC1(C)C'
        assert dataset.df.shape == (4, 1827)

    def test_smiles_dataset_mordred(self):
        dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=MordredDescriptors(ignore_3D=True))
        dataset.create()
        assert dataset.featurizer_name == 'MordredDescriptors'
        assert dataset.smiles_strings[
                   0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
        assert dataset.df.shape == (23, 1614)

    def test_smiles_dataset_rdkit(self):
        dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=RDKitDescriptors())
        dataset.create()
        assert dataset.featurizer_name == 'RDKitDescriptors'
        assert dataset.smiles_strings[
                   0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
        assert dataset.df.shape == (23, 209)

    def test_smiles_dataset_finger(self):
        dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=TopologicalFingerprint())
        dataset.create()
        assert dataset.featurizer_name == 'TopologicalFingerprint'
        assert dataset.smiles_strings[
                   0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
        assert dataset.df.shape == (23, 2049)

    def test_smiles_dataset_molgraph(self):
        dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=MolGraphConvFeaturizer(use_edges=True))
        dataset.create()
        assert dataset.featurizer_name == 'MolGraphConvFeaturizer'
        assert dataset.smiles_strings[
                   0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
        assert dataset.df.shape == (23, 2)

    def test_smiles_torch_dataset(self):
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification')
        dataset.create()

        dataloader = DataLoader(dataset, batch_size=4,
                                shuffle=True, num_workers=0)
        for data in dataloader:
            print(data)

        assert dataset.featurizer_name == 'MolGraphConvFeaturizer'
        assert dataset.smiles_strings[
                   0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
        assert len(dataset.df) == 23

    def test_smiles_torch_tab_dataset(self):
        dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=MordredDescriptors(ignore_3D=True),  task='classification')
        dataset.create()
        dataloader = dl(dataset, batch_size=4,
                                shuffle=True, num_workers=0)
        for data in dataloader:
            from torch import Tensor
            assert type(data[0]) == Tensor
            # print(data)
        assert dataset.featurizer_name == 'MordredDescriptors'
        assert dataset.smiles_strings[
                   0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
        assert len(dataset.df) == 23

    def test_smiles_tab_data_save(self):
        dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=TopologicalFingerprint())
        dataset.create()
        dataset.dataset_name = "Smiles_fingerprints"
        dataset.save()
        assert dataset.featurizer_name == 'TopologicalFingerprint'
        assert dataset.smiles_strings[
                   0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
        assert dataset.df.shape == (23, 2049)

    def test_load_dataset(self):
        dataset = SmilesDataset()
        dataset = dataset.load("./Smiles_fingerprints.jdb")
        assert dataset.featurizer_name == 'TopologicalFingerprint'
        assert dataset.smiles_strings[
                   0] == 'O=C1CCCN1Cc1cccc(C(=O)N2CCC(C3CCNC3)CC2)c1'
        assert dataset.df.shape == (23, 2049)

    def test_seq_dataset(self):
        cid = create_char_to_idx(self.mols)
        feat = SmilesToSeq(char_to_idx=cid)
        dataset = SmilesDataset(smiles=self.mols, y=self.ys_regr, featurizer=feat)
        dataset.create()
        dataloader = dl(dataset, batch_size=4,
                                shuffle=True, num_workers=0)
        for data in dataloader:
            from torch import Tensor
            assert type(data[0]) == Tensor

    def test_image_dataset(self):
        feat = SmilesToImage(img_size=80)
        dataset = SmilesDataset(smiles=self.mols, y=self.ys, featurizer=feat)
        dataset.create()
        dataloader = dl(dataset, batch_size=4,
                                shuffle=True, num_workers=0)
        for data in dataloader:
            from torch import Tensor
            assert type(data[0]) == Tensor
