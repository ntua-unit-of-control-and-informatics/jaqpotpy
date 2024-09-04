import unittest

from jaqpotpy.descriptors.molecular.molecule_graph_conv import (
    MolGraphConvFeaturizer,
    PagtnMolGraphFeaturizer,
    AttentiveFPFeaturizer,
)
import torch
import numpy as np


# pylint: disable=no-member
class TestMolGraphConvFeaturizer(unittest.TestCase):
    @unittest.skip(
        "Torch and graphs have not been tested in the current version of jaqpotpy"
    )
    def test_default_featurizer(self):
        smiles = ["C1=CC=CN=C1", "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"]
        featurizer = MolGraphConvFeaturizer()
        graph_feat = featurizer.featurize(smiles)
        assert len(graph_feat) == 2

        # assert "C1=CC=CN=C1"
        assert graph_feat[0].num_nodes == 6
        assert graph_feat[0].num_node_features == 30
        assert graph_feat[0].num_edges == 12

        # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
        assert graph_feat[1].num_nodes == 22
        assert graph_feat[1].num_node_features == 30
        assert graph_feat[1].num_edges == 44

    @unittest.skip(
        "Torch and graphs have not been tested in the current version of jaqpotpy"
    )
    def test_featurizer_with_use_edge(self):
        smiles = ["C1=CC=CN=C1", "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"]
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        graph_feat = featurizer.featurize(smiles)
        assert len(graph_feat) == 2

        # assert "C1=CC=CN=C1"
        assert graph_feat[0].num_nodes == 6
        assert graph_feat[0].num_node_features == 30
        assert graph_feat[0].num_edges == 12
        assert graph_feat[0].num_edge_features == 11

        # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
        assert graph_feat[1].num_nodes == 22
        assert graph_feat[1].num_node_features == 30
        assert graph_feat[1].num_edges == 44
        assert graph_feat[1].num_edge_features == 11

    @unittest.skip(
        "Torch and graphs have not been tested in the current version of jaqpotpy"
    )
    def test_featurizer_with_use_chirality(self):
        smiles = ["C1=CC=CN=C1", "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"]
        featurizer = MolGraphConvFeaturizer(use_chirality=True)
        graph_feat = featurizer.featurize(smiles)
        assert len(graph_feat) == 2

        # assert "C1=CC=CN=C1"
        assert graph_feat[0].num_nodes == 6
        assert graph_feat[0].num_node_features == 32
        assert graph_feat[0].num_edges == 12

        # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
        assert graph_feat[1].num_nodes == 22
        assert graph_feat[1].num_node_features == 32
        assert graph_feat[1].num_edges == 44

    @unittest.skip(
        "Torch and graphs have not been tested in the current version of jaqpotpy"
    )
    def test_featurizer_with_use_partial_charge(self):
        smiles = ["C1=CC=CN=C1", "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"]
        featurizer = MolGraphConvFeaturizer(use_partial_charge=True)
        graph_feat = featurizer.featurize(smiles)
        assert len(graph_feat) == 2

        # assert "C1=CC=CN=C1"
        assert graph_feat[0].num_nodes == 6
        assert graph_feat[0].num_node_features == 31
        assert graph_feat[0].num_edges == 12

        # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
        assert graph_feat[1].num_nodes == 22
        assert graph_feat[1].num_node_features == 31
        assert graph_feat[1].num_edges == 44

    # @unittest.skip(
    #     "Torch and graphs have not been tested in the current version of jaqpotpy"
    # )
    # def test_torch_molgan_graph(self):
    #     smiles = ["C1=CC=CN=C1", "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"]
    #     featurizer = TorchMolGraphConvFeaturizer()
    # for smil in smiles:
    #     mol = Chem.MolFromSmiles(smil)
    #     data = featurizer.featurize(smil)


class TestPagtnMolGraphConvFeaturizer(unittest.TestCase):
    @unittest.skip(
        "Torch and graphs have not been tested in the current version of jaqpotpy"
    )
    def test_default_featurizer(self):
        smiles = ["C1=CC=CN=C1", "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"]
        featurizer = PagtnMolGraphFeaturizer(max_length=5)
        graph_feat = featurizer.featurize(smiles)
        assert len(graph_feat) == 2
        # assert "C1=CC=CN=C1"
        assert graph_feat[0].num_nodes == 6
        assert graph_feat[0].num_node_features == 94
        assert graph_feat[0].num_edges == 36
        assert graph_feat[0].num_edge_features == 42

        # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
        assert graph_feat[1].num_nodes == 22
        assert graph_feat[1].num_node_features == 94
        assert graph_feat[1].num_edges == 484
        assert graph_feat[0].num_edge_features == 42


class TestAttentiveFPFeaturizer(unittest.TestCase):
    @unittest.skip(
        "Torch and graphs have not been tested in the current version of jaqpotpy"
    )
    def test_default_featurizer(self):
        smiles = ["CCO"]
        featurizer = AttentiveFPFeaturizer(use_loops=False)
        graph_feat = featurizer.featurize(smiles)
        assert len(graph_feat) == len(smiles)

        res_as_dict = {
            "edge_index": torch.LongTensor([[0, 2, 2, 1], [2, 0, 1, 2]]),
            "node_features": torch.FloatTensor(
                [
                    [
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                ]
            ),
            "edge_features": torch.FloatTensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
        }

        # assert Node features
        assert (
            np.array(res_as_dict["node_features"] == graph_feat[0][1][1])
            .flatten()
            .all()
        )

        # assert Edge features
        assert (
            np.array(res_as_dict["edge_features"] == graph_feat[0][2][1])
            .flatten()
            .all()
        )

        print(type(graph_feat[0]))
