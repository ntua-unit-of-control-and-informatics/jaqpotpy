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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=DeprecationWarning)
from jaqpotpy.models import Evaluator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef\
    , precision_score, recall_score, confusion_matrix
import numpy as np


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

    def test_RF(self):
        featurizer = RDKitDescriptors()
        dataset = SmilesDataset(smiles=self.mols, y=self.ys, task='classification', featurizer=featurizer)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        molecularModel_t1.Y = 'DILI'
        from jaqpotpy import Jaqpot
        jaqpot = Jaqpot("http://localhost:8080/jaqpot/services/")
        jaqpot.request_key('jasonsoti1@gmail.com', 'PX-E850E')
        molecularModel_t1.deploy_on_jaqpot(jaqpot=jaqpot,
                                     description="Test AD Model",
                                     model_title="TEST Model")
        # assert molecularModel_t1.doa.IN == [True]

    def test_predict_proba(self):
        featurizer = RDKitDescriptors()
        dataset = SmilesDataset(smiles=self.mols, y=self.ys, task='classification', featurizer=featurizer)
        model = SVC(probability=True)
        molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
        molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
        molecularModel_t1.probability

    def test_eval(self):
        featurizer = RDKitDescriptors()
        dataset = SmilesDataset(smiles=self.mols, y=self.ys, task='regression', featurizer=featurizer)
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

        molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=val).fit()
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
    #     molecularModel_t1 = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
    #     molecularModel_t1('COc1ccc2c(N)nn(C(=O)Cc3cccc(Cl)c3)c2c1')
    #     assert molecularModel_t1.doa.IN == [False]
    #     assert int(molecularModel_t1.doa.doa_new[0]) == 271083
    #     assert int(molecularModel_t1.prediction[0][0]) == -2873819


    def test_ALL_regression_ONNX(self):
        from sklearn.utils import all_estimators
        import onnxruntime as rt

        featurizer = TopologicalFingerprint()
        dataset = SmilesDataset(smiles=self.mols, y=self.ys_regr, task='regression', featurizer=featurizer)

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
                    molecular_model = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()

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

    def test_ONE_regression_ONNX(self):
        from sklearn.utils import all_estimators
        import onnxruntime as rt

        featurizer = RDKitDescriptors()
        dataset = SmilesDataset(smiles=self.mols, y=self.ys_regr, task='regression', featurizer=featurizer)

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.linear_model import LinearRegression, GammaRegressor, ARDRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor

        model = ARDRegression()
        molecular_model = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()
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

    def test_ALL_classification_ONNX(self):
        from sklearn.utils import all_estimators
        import onnxruntime as rt

        featurizer = TopologicalFingerprint()
        dataset = SmilesDataset(smiles=self.mols, y=self.ys, task='classification', featurizer=featurizer)

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
                    molecular_model = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=None).fit()

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

    def test_ONE_classification_ONNX(self):
        from sklearn.utils import all_estimators
        import onnxruntime as rt

        featurizer = TopologicalFingerprint()
        dataset = SmilesDataset(smiles=self.mols, y=self.ys, task='classification', featurizer=featurizer)
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
        molecular_model = MolecularSKLearn(dataset=dataset, doa=Leverage(), model=model, eval=val).fit()
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
