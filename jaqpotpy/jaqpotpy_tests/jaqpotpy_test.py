import unittest
from jaqpotpy import Jaqpot
import pandas as pd
from jaqpotpy.models import MolecularModel


class TestModels(unittest.TestCase):

    @unittest.skip("Tests need refactoring")
    def test_login(self):
        jaqpot = Jaqpot()
        jaqpot.login("", "")

    @unittest.skip("Tests need refactoring")
    def test_fail_login(self):
        jaqpot = Jaqpot()
        jaqpot.login("", "")

    @unittest.skip("Tests need refactoring")
    def test_get_model(self):
        jaqpot = Jaqpot()
        jaqpot.login("", "")
        model = jaqpot.get_model_by_id("Yon4zUuavqY46Umu9AKp")

    @unittest.skip("Tests need refactoring")
    def test_load_mol_mod(self):
        jaqpot = Jaqpot()
        jaqpot.set_api_key("")
        model = MolecularModel().load_from_jaqpot(jaqpot, "id")

    @unittest.skip("Tests need refactoring")
    def test_predict(self):
        data = {'LFG': [0.1, 0.2], 'EQP': [0.1, 0.2], 'NEQ': [0.1, 0.2], 'GAP': [0.1, 0.2]}
        df = pd.DataFrame.from_dict(data)
        jaqpot = Jaqpot()
        jaqpot.login("", "")
        df, predicts = jaqpot.predict(df, "Yon4zUuavqY46Umu9AKp")
        print(df)
        print(predicts)
