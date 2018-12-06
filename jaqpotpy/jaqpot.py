import jaqpotpy.api.login as jaqlogin
import jaqpotpy.api.algorithms_api as alapi
import jaqpotpy.api.dataset_api as data_api
import jaqpotpy.api.models_api as models_api
import jaqpotpy.helpers.jwt as jwtok
import jaqpotpy.helpers.helpers as help
import json
# import simplejson as json
import jaqpotpy.api.feature_api as featapi
from jaqpotpy.helpers.serializer import JaqpotSerializer
from tornado.ioloop import IOLoop
from jaqpotpy.entities.dataset import Dataset
from jaqpotpy.entities.meta import MetaInfo
from jaqpotpy.entities.featureinfo import FeatureInfo
import pandas as pd
import numpy as np

ENCODING = 'utf-8'

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


class Jaqpot:

    def __init__(self, base_url):
        self.base_url = base_url
        self.api_key = None
        self.user_id = None

    def login(self, username, password):
        jaqlogin.authenticate_sync(self.base_url, username=username, password=password)

    def set_api_key(self, api_key):
        self.api_key = api_key
        print("api key is set")

    def request_key(self, username, password):
        try:
            au_req = jaqlogin.authenticate_sync(self.base_url, username, password)
            self.api_key = au_req.authToken
            self.user_id = jwtok.decode_jwt(self.api_key).get('sub')
        except Exception as e:
            print("Error: " + str(e))

    def request_key_safe(self):
        try:
            au_req = jaqlogin.authenticate_sync_hidepass(self.base_url)
            self.api_key = au_req.authToken
            self.user_id = jwtok.decode_jwt(self.api_key).get('sub')
        except Exception as e:
            print("Error: " + str(e))

    def my_info(self):
        print(jwtok.decode_jwt(self.api_key))
        return jwtok.decode_jwt(self.api_key)

    def get_algorithms(self, start=None, max=None):
        self.check_key()
        try:
            algos = alapi.get_allgorithms_sync(self.base_url, self.api_key, start, max)
            return algos
        except Exception as e:
            print("Error:" + str(e))

    def get_algorithms_classes(self, start=None, max=None):
        self.check_key()
        try:
            algos = alapi.get_allgorithms_classes(self.base_url, self.api_key, start, max)
            return algos
        except Exception as e:
            print("Error:" + str(e))

    def upload_dataset(self, df=None, id=None, title=None):
        # print(df.to_json())
        df_titles = list(df)
        for t in df_titles:
            type_to_c = df[t].dtypes
            if type_to_c == 'int64':
                # print(type_to_c)
                df[t] = df[t].astype(float)
        # print(df.dtypes)
        df = df.replace(np.nan, '', regex=True)
        if type(df).__name__ is not 'DataFrame':
            raise Exception("Cannot form a Jaqpot Dataset. Please provide a Dataframe")
        if id is not None:
            df.set_index(id, inplace=True)
            feat_titles = list(df)
            feats = []
            for f_t in feat_titles:
                fe = help.create_feature(f_t, self.user_id)
                jsonmi = json.dumps(fe, cls=JaqpotSerializer)
                # jsonmi = json.dumps(fe.__dict__)
                feats.append(jsonmi)
        else:
            feat_titles = list(df)
            feats = []
            for f_t in feat_titles:
                fe = help.create_feature(f_t, self.user_id)
                # jsonmi = json.dumps(fe.__dict__)
                jsonmi = json.dumps(fe, cls=JaqpotSerializer)
                feats.append(jsonmi)
        feat_map = {}
        featutes = []
        for feat in feats:
            feat_info = FeatureInfo()
            f = IOLoop.current().run_sync(lambda: featapi.create_feature_async(self.base_url, self.api_key, feat))
            feat_uri = self.base_url + "feature/" + f["_id"]
            feat_map[f["meta"]["titles"][0]] = feat_uri
            feat_info.uri = feat_uri
            feat_info.name = f["meta"]["titles"][0]
            featutes.append(feat_info.__dict__)
        dataset = Dataset()
        meta = MetaInfo()
        meta.creators = [self.user_id]
        if title is not None:
            meta.titles = [title]
        meta.descriptions = ["Dataset uploaded from python client"]
        dataset.meta = meta.__dict__
        dataset.totalRows = df.shape[0]
        dataset.totalColumns = df.shape[1]
        data_entry = help.create_data_entry(df, feat_map, self.user_id)
        dataset.dataEntry = data_entry
        dataset.features = featutes
        # jsondataset = json.dumps(dataset)
        jsondataset = json.dumps(dataset, cls=JaqpotSerializer)
        print(jsondataset)
        dataset_n = data_api.create_dataset_sync(self.base_url, self.api_key, jsondataset)
        print("Dataset created with id: " + dataset_n["_id"])

    def deploy_linear_model(self, model, X, y, title, description, algorithm):
        if isinstance(X, pd.DataFrame) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. X is not one')
        if isinstance(y, pd.DataFrame) is False and isinstance(y, pd.Series) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. Y is not one')
        coef_flatten = model.coef_.flatten()
        intercept_flatten = model.intercept_.flatten()
        coef_df = pd.DataFrame(list(zip(list(X), coef_flatten)), columns=['Features', 'coeff'])
        intercept = pd.DataFrame([('(intercept)', intercept_flatten[0])], columns=['Features', 'coeff'])
        coefs_all = intercept.append(coef_df, ignore_index=True)
        coefs = {}
        additionalInfo = {}
        for key in coefs_all.values:
            coefs[key[0]] = key[1]
        additionalInfo['coefficients'] = coefs
        additionalInfo['inputSeries'] = list(X)
        pretrained = help.create_pretrain_req(model, X, y, title, description,
                                              algorithm, "Linear model Scikit learn", "scikit-learn-linear-model",
                                              additionalInfo)
        # j = json.dumps(pretrained)
        j = json.dumps(pretrained, cls=JaqpotSerializer)
        response = models_api.post_pretrained_model(self.base_url, self.api_key, j)
        print("Model with id: " + response['modelId'] + " created. Please visit https://app.jaqpot.org/")

    def deploy_tree(self, model, X, y, title, description, algorithm):
        if isinstance(X, pd.DataFrame) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. X is not one')
        if isinstance(y, pd.DataFrame) is False and isinstance(y, pd.Series) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. Y is not one')
        additionalInfo = {}
        additionalInfo['inputSeries'] = list(X)
        pretrained = help.create_pretrain_req(model, X, y, title, description,
                                              algorithm, "Decision tree Scikit learn", "scikit-learn-tree-model",
                                              additionalInfo)
        j = json.dumps(pretrained, cls=JaqpotSerializer)
        response = models_api.post_pretrained_model(self.base_url, self.api_key, j)
        print("Model with id: " + response['modelId'] + " created. Please visit https://app.jaqpot.org/")

    def deploy_ensemble(self, model, X, y, title, description, algorithm):
        if isinstance(X, pd.DataFrame) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. X is not one')
        if isinstance(y, pd.DataFrame) is False and isinstance(y, pd.Series) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. Y is not one')
        additionalInfo = {}
        additionalInfo['inputSeries'] = list(X)
        pretrained = help.create_pretrain_req(model, X, y, title, description,
                                              algorithm, "Ensemble Scikit learn", "scikit-learn-ensemble-model",
                                              additionalInfo)
        j = json.dumps(pretrained, cls=JaqpotSerializer)
        response = models_api.post_pretrained_model(self.base_url, self.api_key, j)
        print("Model with id: " + response['modelId'] + " created. Please visit https://app.jaqpot.org/")

    def deploy_svm(self, model, X, y, title, description, algorithm):
        if isinstance(X, pd.DataFrame) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. X is not one')
        if isinstance(y, pd.DataFrame) is False and isinstance(y, pd.Series) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. Y is not one')
        additionalInfo = {}
        additionalInfo['inputSeries'] = list(X)
        pretrained = help.create_pretrain_req(model, X, y, title, description,
                                              algorithm, "Svm Scikit learn", "scikit-learn-svm-model",
                                              additionalInfo)
        j = json.dumps(pretrained, cls=JaqpotSerializer)
        response = models_api.post_pretrained_model(self.base_url, self.api_key, j)
        print("Model with id: " + response['modelId'] + " created. Please visit https://app.jaqpot.org/")


    def api_key(self):
        return self.api_key

    def check_key(self):
        if self.api_key is None:
            raise Exception("Api key not present!")
