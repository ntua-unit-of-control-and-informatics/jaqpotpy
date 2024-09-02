import unittest

# import pytest
import numpy as np
from jaqpotpy.descriptors.graph.graph_data import GraphData, BatchGraphData


class TestGraph(unittest.TestCase):
    @unittest.skip(
        "Material modelling has not been tested yet in the newest version of jaqpotpy"
    )
    # @pytest.mark.torch
    def test_graph_data(self):
        num_nodes, num_node_features = 5, 32
        num_edges, num_edge_features = 6, 32
        node_features = np.random.random_sample((num_nodes, num_node_features))
        edge_features = np.random.random_sample((num_edges, num_edge_features))
        edge_index = np.array(
            [
                [0, 1, 2, 2, 3, 4],
                [1, 2, 0, 3, 4, 0],
            ]
        )
        node_pos_features = None

        graph = GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_pos_features=node_pos_features,
        )

        assert graph.num_nodes == num_nodes
        assert graph.num_node_features == num_node_features
        assert graph.num_edges == num_edges
        assert graph.num_edge_features == num_edge_features

        # check convert function
        pyg_graph = graph.to_pyg_graph()
        from torch_geometric.data import Data

        assert isinstance(pyg_graph, Data)

        dgl_graph = graph.to_dgl_graph()
        from dgl import DGLGraph

        assert isinstance(dgl_graph, DGLGraph)

    # @pytest.mark.torch
    @unittest.skip(
        "Material modelling has not been tested yet in the newest version of jaqpotpy"
    )
    def test_invalid_graph_data(self):
        with self.assertRaises(ValueError):
            invalid_node_features_type = list(np.random.random_sample((5, 32)))
            edge_index = np.array(
                [
                    [0, 1, 2, 2, 3, 4],
                    [1, 2, 0, 3, 4, 0],
                ]
            )
            _ = GraphData(
                node_features=invalid_node_features_type,
                edge_index=edge_index,
            )

        with self.assertRaises(ValueError):
            node_features = np.random.random_sample((5, 32))
            invalid_edge_index_shape = np.array(
                [
                    [0, 1, 2, 2, 3, 4],
                    [1, 2, 0, 3, 4, 5],
                ]
            )
            _ = GraphData(
                node_features=node_features,
                edge_index=invalid_edge_index_shape,
            )

        with self.assertRaises(ValueError):
            node_features = np.random.random_sample((5, 5))
            invalid_edge_index_shape = np.array(
                [
                    [0, 1, 2, 2, 3, 4],
                    [1, 2, 0, 3, 4, 0],
                    [2, 2, 1, 4, 0, 3],
                ],
            )
            _ = GraphData(
                node_features=node_features,
                edge_index=invalid_edge_index_shape,
            )

        with self.assertRaises(TypeError):
            node_features = np.random.random_sample((5, 32))
            _ = GraphData(node_features=node_features)

    # @pytest.mark.torch
    @unittest.skip(
        "Material modelling has not been tested yet in the newest version of jaqpotpy"
    )
    def test_batch_graph_data(self):
        num_nodes_list, num_edge_list = [3, 4, 5], [2, 4, 5]
        num_node_features, num_edge_features = 32, 32
        edge_index_list = [
            np.array([[0, 1], [1, 2]]),
            np.array([[0, 1, 2, 3], [1, 2, 0, 2]]),
            np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
        ]

        graph_list = [
            GraphData(
                node_features=np.random.random_sample(
                    (num_nodes_list[i], num_node_features)
                ),
                edge_index=edge_index_list[i],
                edge_features=np.random.random_sample(
                    (num_edge_list[i], num_edge_features)
                ),
                node_pos_features=None,
            )
            for i in range(len(num_edge_list))
        ]
        batch = BatchGraphData(graph_list)

        assert batch.num_nodes == sum(num_nodes_list)
        assert batch.num_node_features == num_node_features
        assert batch.num_edges == sum(num_edge_list)
        assert batch.num_edge_features == num_edge_features
        assert batch.graph_index.shape == (sum(num_nodes_list),)
