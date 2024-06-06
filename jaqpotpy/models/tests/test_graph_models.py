from jaqpotpy.models import MolecularTorchGeometric, GCN_V1, AttentiveFPModel_V1\
    , GhebConvV1, GraphConv_V1, GatedGraphConv_V1, ResGatedGraphConv_V1, GATConv_V1, AttentiveFP
from jaqpotpy.models.torch_models import GCN_V1, SAGEConv_V1
from jaqpotpy.datasets import TorchGraphDataset
from jaqpotpy.models import Evaluator, AttentiveFP_V1
from sklearn.metrics import max_error, mean_absolute_error, r2_score
import torch
from rdkit import Chem
import unittest
from jaqpotpy.descriptors.molecular import MolGraphConvFeaturizer, PagtnMolGraphFeaturizer, AttentiveFPFeaturizer


class TestJitModels(unittest.TestCase):
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

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_tdc_attentive(self):
        import pandas as pd
        from sklearn.metrics import average_precision_score, accuracy_score
        from tdc.benchmark_group import admet_group

        group = admet_group(path='C:/Users/jason/centralenv/TDC/data/')

        benchmark = group.get('CYP3A4_Substrate_CarbonMangels')
        name = benchmark['name']
        train_val = benchmark['train_val']
        test = benchmark['test']

        featurizer = AttentiveFPFeaturizer()

        # featurizer = PagtnMolGraphFeaturizer()

        train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=42)

        train_dataset = TorchGraphDataset(smiles=train['Drug'], y=train['Y'], task='classification'
                                          , featurizer=featurizer)

        test_dataset = TorchGraphDataset(smiles=valid['Drug'], y=valid['Y'], task='classification'
                                         , featurizer=featurizer)

        train_dataset.create()
        test_dataset.create()
        val = Evaluator()
        val.dataset = test_dataset
        val.register_scoring_function('Accuracy', accuracy_score)
        val.register_scoring_function('AUPRC', average_precision_score)

        model = AttentiveFP(in_channels=39, hidden_channels=80, out_channels=2, edge_dim=10, num_layers=6,
                            num_timesteps=3).jittable()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=train_dataset
                                    , model_nn=model, eval=val
                                    , train_batch=262, test_batch=200
                                    , epochs=10, optimizer=optimizer, criterion=criterion, device="cpu", test_metric=(accuracy_score, 'maximize'))  # .fit()

        def cross_train_torch(group, nn, name, test_df, task='regression'):
            predictions_list = []

            for seed in [1, 2, 3, 4, 5]:
                predictions = {}

                # Train - Validation split
                train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)

                # Create the Jaqpot Datasets
                jaq_train = TorchGraphDataset(smiles=train['Drug'], y=train['Y'], featurizer=nn.dataset.featurizer,
                                              task=task)
                jaq_train.create()

                jaq_val = TorchGraphDataset(smiles=valid['Drug'], y=valid['Y'], featurizer=nn.dataset.featurizer,
                                            task=task)
                jaq_val.create()

                # Update the datasets
                nn.evaluator.dataset = jaq_val
                nn.dataset = jaq_train

                # Train the model
                trained_model = nn.fit()

                # Create molecular model and take predictions on the Test set
                mol_model = trained_model.create_molecular_model()
                mol_model(test_df['Drug'].tolist())

                # Keep the predictions
                predictions[name] = mol_model.prediction
                predictions_list.append(predictions)

            # Return the cross validation score
            return group.evaluate_many(predictions_list)

        # evaluation = cross_train_torch(group, m, name, test, 'classification')

        # print(evaluation)

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_0(self):
        featurizer = MolGraphConvFeaturizer()
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('MAE', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = GCN_V1(30, 3, 40, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion, test_metric=(mean_absolute_error,'minimize')).fit()

        m.eval()

        model = m.create_molecular_model()
        model(self.mols[0])
        model(self.mols)

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_1(self):
        featurizer = MolGraphConvFeaturizer(use_chirality=True)
        g_data = featurizer.featurize(self.mols[0])
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = GCN_V1(32, 3, 40, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        model = m.create_molecular_model()
        model(self.mols[0])
        model(self.mols)

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_2(self):
        featurizer = MolGraphConvFeaturizer(use_partial_charge=True)
        g_data = featurizer.featurize(self.mols[0])
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = GCN_V1(31, 3, 40, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        model = m.create_molecular_model()
        model(self.mols[0])

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_3(self):
        featurizer = PagtnMolGraphFeaturizer()
        g_data = featurizer.featurize("CCN1CCN(C(=O)N[C@@H](C(=O)N[C@@H]2C(=O)N3C(C(=O)O)=C(CSc4nnnn4C)CS[C@H]23)c2ccc(O)cc2)C(=O)C1=O")
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = AttentiveFP(in_channels=94, hidden_channels=40, out_channels=2, edge_dim=42, num_layers=2,
                                    num_timesteps=2).jittable()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        model = m.create_molecular_model()
        model(self.mols[0])

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_4(self):
        featurizer = AttentiveFPFeaturizer()
        g_data = featurizer.featurize(self.mols[0])
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = AttentiveFP(in_channels=39, hidden_channels=40, out_channels=39, edge_dim=10, num_layers=2,
                                    num_timesteps=2).jittable()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        model = m.create_molecular_model()
        model(self.mols[0])
        model(self.mols)

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_5(self):
        featurizer = MolGraphConvFeaturizer()
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = GCN_V1(30, 3, 40, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        model = m.create_molecular_model()
        model(self.mols[0])
        model(self.mols)

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_6(self):
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = GCN_V1(30, 3, 40, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        try:
            m = MolecularTorchGeometric(dataset=dataset
                                        , model_nn=model, eval=val
                                        , train_batch=4, test_batch=4
                                        , epochs=40, optimizer=optimizer, criterion=criterion).fit()
            model = m.create_molecular_model()
            model(self.mols[0])
            model(self.mols)
        except TypeError as e:
            print(e)
            pass

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_7(self):
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = AttentiveFP(in_channels=30, hidden_channels=40, out_channels=2, edge_dim=11, num_layers=2,
                                    num_timesteps=2).jittable()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        try:
            m = MolecularTorchGeometric(dataset=dataset
                                        , model_nn=model, eval=val
                                        , train_batch=4, test_batch=4
                                        , epochs=40, optimizer=optimizer, criterion=criterion).fit()
            model = m.create_molecular_model()
            model(self.mols[0])
            model(self.mols)
        except TypeError as e:
            print(e)
            pass
	    
    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_8_regr(self):
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys_regr, task='regression'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = AttentiveFP(in_channels=30, hidden_channels=40, out_channels=1, edge_dim=11, num_layers=2,
                                    num_timesteps=2).jittable()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.L1Loss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        model = m.create_molecular_model()
        model(self.mols[0])
        model(self.mols)

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_9(self):
        featurizer = MolGraphConvFeaturizer(use_edges=True, use_chirality=True)
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = AttentiveFP(in_channels=32, hidden_channels=40, out_channels=2, edge_dim=11, num_layers=2,
                                    num_timesteps=2).jittable()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        try:
            m = MolecularTorchGeometric(dataset=dataset
                                        , model_nn=model, eval=val
                                        , train_batch=4, test_batch=4
                                        , epochs=40, optimizer=optimizer, criterion=criterion).fit()
            model = m.create_molecular_model()
            model(self.mols[0])
            model(self.mols)
        except TypeError as e:
            print(e)
            pass

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_10(self):
        featurizer = MolGraphConvFeaturizer()
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = SAGEConv_V1(30, 3, 40, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        model = m.create_molecular_model()
        model(self.mols[0])
        model(self.mols)

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_11(self):
        featurizer = MolGraphConvFeaturizer()
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = GhebConvV1(30, 3, 40, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        model = m.create_molecular_model()
        model(self.mols[0])
        model(self.mols)

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_12(self):
        featurizer = MolGraphConvFeaturizer()
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = GraphConv_V1(30, 3, 40, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        model = m.create_molecular_model()
        model(self.mols[0])
        model(self.mols)

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_13(self):
        featurizer = MolGraphConvFeaturizer()
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = GatedGraphConv_V1(30, 3, 40, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        model = m.create_molecular_model()
        model(self.mols[0])
        model(self.mols)

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_14(self):
        featurizer = MolGraphConvFeaturizer()
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = GatedGraphConv_V1(30, 3, 40, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        model = m.create_molecular_model()
        model(self.mols[0])
        model(self.mols)
	    
    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_15(self):
        featurizer = MolGraphConvFeaturizer()
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = ResGatedGraphConv_V1(30, 3, 40, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        model = m.create_molecular_model()
        model(self.mols[0])
        model(self.mols)

    @unittest.skip("Torch and graphs have not been tested in the current version of jaqpotpy")
    def test_torch_graph_models_16(self):
        featurizer = MolGraphConvFeaturizer()
        dataset = TorchGraphDataset(smiles=self.mols, y=self.ys, task='classification'
                                    , featurizer=featurizer)
        dataset.create()
        val = Evaluator()
        val.dataset = dataset
        val.register_scoring_function('Max Error', max_error)
        val.register_scoring_function('Mean Absolute Error', mean_absolute_error)
        val.register_scoring_function('R 2 score', r2_score)
        model = GATConv_V1(30, 3, 40, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        m = MolecularTorchGeometric(dataset=dataset
                                    , model_nn=model, eval=val
                                    , train_batch=4, test_batch=4
                                    , epochs=40, optimizer=optimizer, criterion=criterion).fit()
        model = m.create_molecular_model()
        model(self.mols[0])
        model(self.mols)

