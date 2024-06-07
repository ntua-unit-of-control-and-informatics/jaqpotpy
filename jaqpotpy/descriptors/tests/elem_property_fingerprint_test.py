import unittest
from jaqpotpy.descriptors.material import ElementPropertyFingerprint

class test_ElementPropertyFingerprint(unittest.TestCase):

    def setUp(self) -> None:
        # self.comp = "Fe2O3"
        self.comp = ['Fe2O3', 'FeO']
        # df = pd.read_csv('C:/Users/jason/centralenv/LTKB/dili_formula.csv')
        # self.comp = list(df['Formula'])

        self.featurizer = ElementPropertyFingerprint()

    @unittest.skip("Material modelling has not been tested yet in the newest version of jaqpotpy")
    def test_featurizer(self):

        features = self.featurizer.featurize('FeO')
        assert features[0].shape == (65,)
        return

    @unittest.skip("Material modelling has not been tested yet in the newest version of jaqpotpy")
    def test_featurizer_df(self):
        features = self.featurizer.featurize_dataframe(self.comp)
        stats = ["minimum", "maximum", "range", "mean", "std_dev"]
        feats = [
            "X",
            "row",
            "group",
            "block",
            "atomic_mass",
            "atomic_radius",
            "mendeleev_no",
            "electrical_resistivity",
            "velocity_of_sound",
            "thermal_conductivity",
            "melting_point",
            "bulk_modulus",
            "coefficient_of_linear_thermal_expansion",
        ]
        assert (features.columns == [i + '_' + j for j in feats for i in stats]).all()
        if isinstance(self.comp, str):
            rows = 1
        else:
            rows = len(self.comp)
        assert features.shape == (rows,len(stats)*len(feats))

        return

if __name__ == '__main__':
    unittest.main()