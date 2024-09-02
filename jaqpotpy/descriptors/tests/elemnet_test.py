from jaqpotpy.descriptors.material import ElementNet
import unittest
import pandas as pd
import numpy as np


class test_ElementNet(unittest.TestCase):
    def setUp(self) -> None:
        self.comp = ["Fe2O3", "FeO"]
        self.featurizer = ElementNet()

    @unittest.skip(
        "Material modelling has not been tested yet in the newest version of jaqpotpy"
    )
    def test_featurize(self):
        features = self.featurizer.featurize(self.comp)
        assert isinstance(features[0], np.ndarray)
        assert len(features[0]) == 118
        assert len(features) == 2
        return

    @unittest.skip(
        "Material modelling has not been tested yet in the newest version of jaqpotpy"
    )
    def test_featurize_df(self):
        features = self.featurizer.featurize_dataframe(self.comp)
        assert isinstance(features, pd.DataFrame)
        assert features.shape == (2, 118)
        return


if __name__ == "__main__":
    unittest.main()
