"""
Tests for Jaqpotpy Datasets.
"""
import unittest
from jaqpotpy.datasets import CompositionDataset, StructureDataset
from jaqpotpy.descriptors.material import *
from pymatgen.core.structure import Structure, Lattice


class TestMatDatasets(unittest.TestCase):
    def setUp(self) -> None:

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
        self.structs_dict = [{'@module': 'pymatgen.core.structure',
                              '@class': 'Structure',
                              'charge': None,
                              'lattice': {'matrix': [[4.2, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 4.2]],
                               'a': 4.2,
                               'b': 4.2,
                               'c': 4.2,
                               'alpha': 90.0,
                               'beta': 90.0,
                               'gamma': 90.0,
                               'volume': 74.08800000000001},
                              'sites': [{'species': [{'element': 'Cs', 'occu': 1}],
                                'abc': [0.0, 0.0, 0.0],
                                'xyz': [0.0, 0.0, 0.0],
                                'label': 'Cs',
                                'properties': {}},
                               {'species': [{'element': 'Cl', 'occu': 1}],
                                'abc': [0.5, 0.5, 0.5],
                                'xyz': [2.1, 2.1, 2.1],
                                'label': 'Cl',
                                'properties': {}}]},
                             {'@module': 'pymatgen.core.structure',
                              '@class': 'Structure',
                              'charge': None,
                              'lattice': {'matrix': [[5.3, 0.0, 0.0], [0.0, 5.3, 0.0], [0.0, 0.0, 5.3]],
                               'a': 5.3,
                               'b': 5.3,
                               'c': 5.3,
                               'alpha': 90.0,
                               'beta': 90.0,
                               'gamma': 90.0,
                               'volume': 148.87699999999998},
                              'sites': [{'species': [{'element': 'Cl', 'occu': 1}],
                                'abc': [0.0, 0.0, 0.0],
                                'xyz': [0.0, 0.0, 0.0],
                                'label': 'Cl',
                                'properties': {}},
                               {'species': [{'element': 'Ca', 'occu': 1}],
                                'abc': [0.6, 0.6, 0.6],
                                'xyz': [3.1799999999999997, 3.1799999999999997, 3.1799999999999997],
                                'label': 'Ca',
                                'properties': {}}]},
                             {'@module': 'pymatgen.core.structure',
                              '@class': 'Structure',
                              'charge': None,
                              'lattice': {'matrix': [[6.7, 0.0, 0.0], [0.0, 6.7, 0.0], [0.0, 0.0, 6.7]],
                               'a': 6.7,
                               'b': 6.7,
                               'c': 6.7,
                               'alpha': 90.0,
                               'beta': 90.0,
                               'gamma': 90.0,
                               'volume': 300.76300000000003},
                              'sites': [{'species': [{'element': 'Ca', 'occu': 1}],
                                'abc': [0.0, 0.0, 0.0],
                                'xyz': [0.0, 0.0, 0.0],
                                'label': 'Ca',
                                'properties': {}},
                               {'species': [{'element': 'Fe', 'occu': 1}],
                                'abc': [0.7, 0.7, 0.7],
                                'xyz': [4.6899999999999995, 4.6899999999999995, 4.6899999999999995],
                                'label': 'Fe',
                                'properties': {}}]},
                             {'@module': 'pymatgen.core.structure',
                              '@class': 'Structure',
                              'charge': None,
                              'lattice': {'matrix': [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 2.5]],
                               'a': 2.5,
                               'b': 2.5,
                               'c': 2.5,
                               'alpha': 90.0,
                               'beta': 90.0,
                               'gamma': 90.0,
                               'volume': 15.625},
                              'sites': [{'species': [{'element': 'Fe', 'occu': 1}],
                                'abc': [0.0, 0.0, 0.0],
                                'xyz': [0.0, 0.0, 0.0],
                                'label': 'Fe',
                                'properties': {}},
                               {'species': [{'element': 'Ag', 'occu': 1}],
                                'abc': [0.8, 0.8, 0.8],
                                'xyz': [2.0, 2.0, 2.0],
                                'label': 'Ag',
                                'properties': {}}]},
                             {'@module': 'pymatgen.core.structure',
                              '@class': 'Structure',
                              'charge': None,
                              'lattice': {'matrix': [[1.9, 0.0, 0.0], [0.0, 1.9, 0.0], [0.0, 0.0, 1.9]],
                               'a': 1.9,
                               'b': 1.9,
                               'c': 1.9,
                               'alpha': 90.0,
                               'beta': 90.0,
                               'gamma': 90.0,
                               'volume': 6.858999999999999},
                              'sites': [{'species': [{'element': 'Ag', 'occu': 1}],
                                'abc': [0.0, 0.0, 0.0],
                                'xyz': [0.0, 0.0, 0.0],
                                'label': 'Ag',
                                'properties': {}},
                               {'species': [{'element': 'Cs', 'occu': 1}],
                                'abc': [0.9, 0.9, 0.9],
                                'xyz': [1.71, 1.71, 1.71],
                                'label': 'Cs',
                                'properties': {}}]}]
        self.path = 'jaqpotpy/test_data/comp_dataset_data.csv'

        self.ys = [
            0, 1, 1, 1, 0
        ]

        self.ys_regr = [
            0.001, 1.286, 2.8756, 1.021, 1.265,
        ]

        # self.path = ''

    @unittest.skip("Material modelling has not been tested yet in the newest version of jaqpotpy")
    def test_composition_manual(self):
        featurizer = ElementNet()
        dataset = CompositionDataset(compositions=self.compositions, y=self.ys_regr, task='regression', featurizer=featurizer)
        dataset.create()
        # print(dataset.materials)
        return

    @unittest.skip("Material modelling has not been tested yet in the newest version of jaqpotpy")
    def test_composition_csv(self):
        featurizer = ElementNet()
        dataset = CompositionDataset(path=self.path, compositions='Comps', keep_cols=['Feature1','Feature2'], y_cols=['class','reg'], task='regression', featurizer=featurizer)
        dataset.create()
        assert 'Feature1' in dataset.df.columns
        return
    
    @unittest.skip("Material modelling has not been tested yet in the newest version of jaqpotpy")
    def test_structure_from_structs(self):
        featurizer = SineCoulombMatrix()
        structs = [
            Structure(self.lattices[i], [self.atoms[i], self.atoms[i + 1]], self.coords[i])
            for i in range(5)
        ]
        dataset = StructureDataset(structures=structs, y=self.ys_regr, featurizer=featurizer)
        dataset.create()
        assert 'Feat_0' in dataset.df.columns
        assert 'Endpoint' in dataset.df.columns

    @unittest.skip("Material modelling has not been tested yet in the newest version of jaqpotpy")
    def test_structure_from_dict(self):
        featurizer = SineCoulombMatrix()
        dataset = StructureDataset(structures=[{
            'lattice': self.lattices[i],
            'species': [self.atoms[i], self.atoms[i + 1]],
            'coords': self.coords[i]} for i in range(5)], y=self.ys_regr, featurizer=featurizer)
        dataset.create()
        # print(dataset.df)
        assert 'Feat_0' in dataset.df.columns
        assert 'Endpoint' in dataset.df.columns

if __name__ == '__main__':
    unittest.main()