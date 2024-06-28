"""
Tests for sklearn models through the jaqpotpy module.
"""
import unittest
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,\
                            precision_score, recall_score, confusion_matrix

from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from jaqpotpy.descriptors.molecular import MordredDescriptors,RDKitDescriptors,TopologicalFingerprint
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.models import SklearnModel
from jaqpotpy.doa.doa import Leverage
from jaqpotpy.models import Evaluator


# Add the following tests:
# 1.regression model
#  1.a. x and y preprocessed 
#  1.b. only x preprocessed 
#  1.c. only y preprocessed 
#  1.d. no preprocessesing 
# 1.Classification model
#  1.a. x and y preprocessed 
#  1.b. only x preprocessed 
#  1.c. only y preprocessed 
#  1.d. no preprocessesing 

class TestModels(unittest.TestCase):
    """
    TestModels is a unit testing class for validating various machine learning models applied to molecular datasets.
    It uses the unittest framework to run tests on classification and regression tasks, evaluating model performance
    and ensuring correct implementation.

    Attributes:
        smiles (list): A list of SMILES strings representing molecular structures used for testing.
        activity (list): A list of binary classification labels corresponding to the molecules in mols.
        classification_dataset (list): A list of continuous regression targets corresponding to the molecules in mols.
        regression_dataset (list): A list of continuous regression targets corresponding to the molecules in mols.
        X_train, X_test, y_train, y_test  (list): A list of continuous regression targets corresponding to the molecules in mols.
        y_test (list): A list of continuous regression targets corresponding to the molecules in mols.

    Methods:
        test_rf_classification(): Tests a RandomForestClassifier on a molecular dataset with 
                                          Mordred fingerprints for classification.
        test_rf_classification(): Tests a RandomForestRegressor on a molecular dataset with 
                                          Mordred fingerprints for regression.                                   
        test_predict_proba(): Tests the probability prediction of an SVC model using RDKit descriptors for classification.
        test_eval(): Tests the evaluation of an SVC model using various scoring functions for regression.
        test_ALL_regression_ONNX(): Tests all available regression models in scikit-learn for ONNX compatibility.
        test_ONE_regression_ONNX(): Tests a specific regression model (ARDRegression) for ONNX compatibility.
        test_ALL_classification_ONNX(): Tests all available classification models in scikit-learn for ONNX compatibility.
        test_ONE_classification_ONNX(): Tests a specific classification model (QuadraticDiscriminantAnalysis) for ONNX compatibility.
    """
    def setUp(self) -> None:

        self.mols = []
        self.ys = []
        self.ys_regr = []


        script_dir = os.path.dirname(__file__)
        test_data_dir = os.path.abspath(os.path.join(script_dir, '../../test_data'))
        clasification_csv_file_path = os.path.join(test_data_dir, 'test_data_smiles_classification.csv')
        regression_csv_file_path = os.path.join(test_data_dir, 'test_data_smiles_regression.csv')
        self.classification_df = pd.read_csv(clasification_csv_file_path)
        self.regression_df = pd.read_csv(regression_csv_file_path)


        #self.smiles = 
        #self.activity = 
        #self.featurizer =  MordredDescriptors()
        #self.classification_dataset = SmilesDataset(smiles=self.smiles, y=self.activity, task='classification', featurizer=self.featurizer)
        #self.regression_dataset = SmilesDataset(smiles=self.smiles, y=self.activity, task='regression', featurizer=self.featurizer)
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        #    self.X, self.y, test_size=0.2, random_state=42)

    @unittest.skip("This test needs refactoring")
    # def test_random_forest_fit_predict(self)->None:
    #     """
    #     This test verifies that the RandomForestClassifier can fit a model to the training data and
    #     make predictions on the test data. It checks that the number of predictions matches the number
    #     of test samples and that the accuracy is above a certain threshold (in this case, 0.7).
    #     """
    #     # Train the model
    #     self.model.fit(self.X_train, self.y_train)
        
    #     # Make predictions
    #     y_pred = self.model.predict(self.X_test)
        
    #     # Check if predictions have the same length as the test set
    #     self.assertEqual(len(y_pred), len(self.y_test), 
    #                      "The number of predictions does not match the number of test samples.")
        
    #     # Check if the model accuracy is within an acceptable range
    #     accuracy = accuracy_score(self.y_test, y_pred)
    #     self.assertGreater(accuracy, 0.2, "The accuracy is lower than expected.")
    
    # @unittest.skip("This test needs refactoring")
    # def test_random_forest_consistency(self):
    #     # Train the model and make predictions
    #     self.model.fit(self.X_train, self.y_train)
    #     y_pred1 = self.model.predict(self.X_test)

    #     # Train the model again and make predictions
    #     self.model.fit(self.X_train, self.y_train)
    #     y_pred2 = self.model.predict(self.X_test)

    #     # Check if predictions are consistent across runs
    #     self.assertListEqual(list(y_pred1), list(y_pred2), 
    #                          "Predictions are not consistent across different training runs.")

    @unittest.skip("This test needs refactoring")
    def test_random_forest(self):
        """
        Test RandomForestClassifier on a molecular dataset with MACCSKeys fingerprints for classification.
        """
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        molecularModel_t1 = SklearnModel(dataset=None, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        molecularModel_t1.Y = 'DILI'
        assert molecularModel_t1.doa.in_doa == [True]

    @unittest.skip("This needs refactoring, as it is it doesn't test anything")
    def test_predict_proba(self):
        """
        Test the probability prediction of an SVC model using RDKit descriptors for classification.
        """
        featurizer = RDKitDescriptors()
        dataset = JaqpotpyDataset(smiles=self.mols, y=self.ys, task='classification', featurizer=featurizer)
        model = SVC(probability=True)
        molecularModel_t1 = SklearnModel(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        molecularModel_t1.probability

    @unittest.skip("This test needs refactoring")
    def test_eval(self):
        """
        Test the evaluation of an SVC model using various scoring functions for regression.
        """
        featurizer = RDKitDescriptors()
        dataset = JaqpotpyDataset(smiles=self.mols, y=self.ys, task='regression', featurizer=featurizer)
        model = SVC(probability=True)

        val = Evaluator()
        val.dataset = dataset

        val.register_scoring_function('Accuracy', accuracy_score)
        val.register_scoring_function('Binary f1', f1_score)
        val.register_scoring_function('Roc Auc', roc_auc_score)
        val.register_scoring_function("MCC", matthews_corrcoef)
        val.register_scoring_function("Precision", precision_score)
        val.register_scoring_function("Recall", recall_score)
        val.register_scoring_function("Confusion Matrix", confusion_matrix)

        molecularModel_t1 = SklearnModel(dataset=dataset, doa=Leverage(), model=model, eval=val).fit()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        print(molecularModel_t1.probability)


    # # This test runs in local due to import of data and not in mem data.
    # def test_SVM(self):
    #     featurizer = MordredDescriptors(ignore_3D=True)
    #     path = './test_data/data.csv'
    #     dataset = MolecularTabularDataset(path=path
    #                                       , x_cols=['molregno', 'organism']
    #                                       , y_cols=['standard_value']
    #                                       , smiles_col='canonical_smiles'
    #                                       , featurizer=featurizer
    #                                       ,
    #                                       X=['nBase', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A',
    #                                          'LogEE_A',
    #                                          'VE1_A', 'VE2_A']
    #                                       )
    #     model = LinearRegression()
    #     molecularModel_t1 = SklearnModel(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
    #     molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
    #     assert molecularModel_t1.doa.in_doa == [False]
    #     assert int(molecularModel_t1.doa.doa_new[0]) == 271083
    #     assert int(molecularModel_t1.prediction[0][0]) == -2873819

    @unittest.skip("This test needs refactoring")
    def test_ALL_regression_ONNX(self):
        """
        Test all available regression models in scikit-learn for ONNX compatibility.
        """
        from sklearn.utils import all_estimators
        import onnxruntime as rt

        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(smiles=self.mols, y=self.ys_regr, task='regression', featurizer=featurizer)

        regression_estimators = all_estimators(type_filter='regressor')

        onnx_exceptions = [
            'CCA',
            'DummyRegressor',
            'KernelRidge',
            'PLSCanonical',
            'PLSRegression',
            'TransformedTargetRegressor'
        ]

        missmatch_predictions = [
            'GaussianProcessRegressor',
            'KNeighborsRegressor',
            'LinearRegression'
        ]

        for name, RegressorClass in regression_estimators:
            if name not in onnx_exceptions and \
               name not in missmatch_predictions and \
               name != 'IsotonicRegression': # Because IsotonicRegression requires only 1 input feature

                try:
                    if name == 'RANSACRegressor':
                        model = RegressorClass(n_samples=15) # n_samples must be less than len(train_df) and 15<23
                    else:
                        model = RegressorClass()
                except Exception:
                    pass
                else:
                    # print(name)
                    molecular_model = SklearnModel(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()

                    new_mols = ['COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
                               , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
                               , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
                               , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21']

                    molecular_model(new_mols)
                    skl_feat = featurizer.featurize_dataframe(new_mols)

                    skl_preds = molecular_model.model.predict(skl_feat)
                    precision = 2
                    try:
                        assert ([round(item[0], precision) for item in molecular_model.prediction] == [round(float(item), precision) for item in skl_preds])
                    except TypeError:
                        # Some models return the predictions as a 2d array size = (2, 1)
                        # These models are GaussianProcessRegressor, KNeighborsRegressor, LinearRegression,
                        # MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso, MultiTaskLassoCV,
                        # RadiusNeighborsRegressor, Ridge and RidgeCV
                        assert ([round(item[0], precision) for item in molecular_model.prediction] == [round(float(item), precision) for item in skl_preds])
    
    @unittest.skip("This test needs refactoring")
    def test_ONE_regression_ONNX(self):
        from sklearn.utils import all_estimators
        import onnxruntime as rt

        featurizer = RDKitDescriptors()
        dataset = JaqpotpyDataset(smiles=self.mols, y=self.ys_regr, task='regression', featurizer=featurizer)

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.linear_model import LinearRegression, GammaRegressor, ARDRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor

        model = ARDRegression()
        molecular_model = SklearnModel(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        sess = rt.InferenceSession(molecular_model.inference_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        new_mols = ['COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
            , 'CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1'
            , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
            , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
            , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21']

        onnx_inp_feat = featurizer.featurize(new_mols)
        pred_onx = sess.run(None, {input_name: onnx_inp_feat.astype(np.float32)})
        molecular_model(new_mols)
        print([round(float(item), 5) for item in pred_onx[0]])
        #
        # try:
        #     print([round(item[0], 5) for item in molecular_model.prediction])
        # except:
        #     print([round(item, 5) for item in molecular_model.prediction])
    
    @unittest.skip("This test needs refactoring")
    def test_ALL_classification_ONNX(self):
        from sklearn.utils import all_estimators
        import onnxruntime as rt

        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(smiles=self.mols, y=self.ys, task='classification', featurizer=featurizer)

        classification_estimators = all_estimators(type_filter='classifier')

        onnx_exceptions = [
            'DummyClassifier',
            'GaussianProcessClassifier', # Weird exception!
            'LabelPropagation',
            'LabelSpreading',
            'NearestCentroid'
        ]

        missmatch_predictions = [
            'KNeighborsClassifier',
            'QuadraticDiscriminantAnalysis'
        ]

        for name, ClassifierClass in classification_estimators:
            if name not in onnx_exceptions and \
               name not in missmatch_predictions and \
               name != 'ComplementNB' and name != 'CategoricalNB' and name!='AdaBoostClassifier':

                try:
                    # if name == 'RANSACRegressor':
                    #     model = ClassifierClass(n_samples=15) # n_samples must be less than len(train_df) and 15<23
                    # else:
                    model = ClassifierClass()
                except Exception:
                    pass
                else:
                    # print(name)
                    molecular_model = SklearnModel(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()

                    new_mols = ['COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
                               , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
                               , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
                               , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21']

                    skl_feat = featurizer.featurize_dataframe(new_mols)
                    skl_preds = molecular_model.model.predict(skl_feat)
                    sk_preds = []
                    for p in skl_preds:
                        sk_preds.append([p])
                    molecular_model(new_mols)
                    precision = 5
                    assert (molecular_model.prediction == sk_preds)
                    # assert [round(item, precision) for item in [item[0] for item in molecular_model.probability]] == [round(float(item), precision) for item in [item[0] for item in pred_onx[1]]]
    
    @unittest.skip("This test needs refactoring")
    def test_ONE_classification_ONNX(self):
        from sklearn.utils import all_estimators
        import onnxruntime as rt

        featurizer = TopologicalFingerprint()
        dataset = JaqpotpyDataset(smiles=self.mols, y=self.ys, task='classification', featurizer=featurizer)
        val = Evaluator()
        val.dataset = dataset

        val.register_scoring_function('Accuracy', accuracy_score)
        val.register_scoring_function('Binary f1', f1_score)
        val.register_scoring_function('Roc Auc', roc_auc_score)
        val.register_scoring_function("MCC", matthews_corrcoef)
        val.register_scoring_function("Precision", precision_score)
        val.register_scoring_function("Recall", recall_score)
        val.register_scoring_function("Confusion Matrix", confusion_matrix)
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.dummy import DummyClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        model = QuadraticDiscriminantAnalysis()
        molecular_model = SklearnModel(dataset=dataset, doa=Leverage(), model=model, eval=val).fit()
        # sess = rt.InferenceSession(molecular_model.inference_model.SerializeToString())
        # input_name = sess.get_inputs()[0].name

        new_mols = ['COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1'
            , 'CNCC1CCCN(C(=O)[C@@H](c2ccccc2)N2Cc3ccccc3C2=O)C1'
            , 'O=C1NC2(CCOc3ccc(Cl)cc32)C(=O)N1c1cncc2ccccc12'
            , 'COc1ccc2c(NC(=O)C3CCOc4ccc(Cl)cc43)[nH]nc2c1'
            , 'O=C(NC1N=Nc2ccccc21)C1CCOc2ccc(Cl)cc21']

        # onnx_inp_feat = featurizer.featurize(new_mols)
        # pred_onx = sess.run(None, {input_name: onnx_inp_feat.astype(np.float32)})
        molecular_model(new_mols)
        # assert len(pred_onx)==2
        # print(pred_onx)
        # print(molecular_model.prediction)
        # print(pred_onx[0])
        # print(pred_onx[0].flatten())

        # print(molecular_model.probability)
        # print(pred_onx[1])


if __name__ == '__main__':
    unittest.main()