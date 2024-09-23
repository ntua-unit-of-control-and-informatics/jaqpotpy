import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from jaqpotpy.descriptors.molecular import TopologicalFingerprint
from jaqpotpy.datasets import JaqpotpyDataset
from jaqpotpy.models import SklearnModel
from jaqpotpy.doa.doa import Leverage, MeanVar, BoundingBox
from jaqpotpy.models.preprocessing import Preprocess
from jaqpotpy import Jaqpot

path = "./jaqpotpy/test_data/test_data_smiles_regression_multioutput.csv"

df = pd.read_csv(path)
smiles_cols = ["SMILES"]
y_cols = ["ACTIVITY"]
x_cols = ["X1", "X2"]
featurizer = TopologicalFingerprint()
dataset = JaqpotpyDataset(
    df=df,
    y_cols=y_cols,
    smiles_cols=smiles_cols,
    x_cols=None,
    task="regression",
    featurizer=featurizer,
)
pre = Preprocess()
pre.register_preprocess_class("Standard Scaler", StandardScaler())
pre.register_preprocess_class_y("Standard Scaler", StandardScaler())

model = RandomForestRegressor(random_state=42)
doa_method = BoundingBox()
molecularModel_t1 = SklearnModel(
    dataset=dataset, doa=None, model=model, evaluator=None, preprocessor=None
)

molecularModel_t1.fit()

# Upload locally
# jaqpot = Jaqpot(base_url="http://localhost.jaqpot.org", app_url="http://localhost.jaqpot.org:3000", 
#                 login_url="http://localhost.jaqpot.org:8070",
#                 api_url="http://localhost.jaqpot.org:8080", keycloak_realm="jaqpot-local", 
#                 keycloak_client_id="jaqpot-local-test")

jaqpot = Jaqpot()
jaqpot.login()
molecularModel_t1.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="DEMO: Only with smiles",
    description=None,
    visibility="PRIVATE",
)
