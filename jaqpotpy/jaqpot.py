import jaqpotpy.api.login as jaqlogin
import jaqpotpy.api.algorithms_api as alapi
import jaqpotpy.api.dataset_api as data_api
import jaqpotpy.api.models_api as models_api
import jaqpotpy.api.task_api as task_api
import jaqpotpy.api.doa_api as doa_api
import jaqpotpy.helpers.jwt as jwtok
import jaqpotpy.helpers.helpers as help
import jaqpotpy.helpers.dataset_deserializer as ds
# from jaqpotpy.helpers.logging import ColoredFormatter
from jaqpotpy.helpers.logging import init_logger
import json
import jaqpotpy.api.feature_api as featapi
from jaqpotpy.helpers.serializer import JaqpotSerializer
# from tornado.ioloop import IOLoop
from jaqpotpy.entities.dataset import Dataset
from jaqpotpy.entities.meta import MetaInfo
from jaqpotpy.entities.featureinfo import FeatureInfo
import pandas as pd
import numpy as np
import http.client as http_client
import getpass
import time
import asyncio
import jaqpotpy.helpers.doa as jha
from collections import namedtuple
import logging

ENCODING = 'utf-8'

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


class Jaqpot:
    """
    Deploys sklearn models on Jaqpot.

    Extended description of function.

    Parameters
    ----------
    base_url : The url on which Jaqpot services are deployed

    """

    def __init__(self, base_url=None, create_logs=False):
        # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        self.log = init_logger(__name__, testing_mode=False, output_log_file=create_logs)
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = ''
        self.api_key = None
        self.user_id = None
        self.http_client = http_client

    def login(self, username, password):
        """
        Logins on Jaqpot.

        Parameters
        ----------
        username : username
        password : password

        """
        try:
            au_req = jaqlogin.authenticate_sync(self.base_url, username, password)
            self.api_key = au_req['authToken']
            self.user_id = jwtok.decode_jwt(self.api_key).get('sub')
        except Exception as e:
            self.log.error("Error: " + str(e))
            # print("Error: " + str(e))

    def set_api_key(self, api_key):
        """
        Set's api key for authentication on Jaqpot.

        Parameters
        ----------
        api_key : api_key can be retireved from the application after logged in

        """
        self.api_key = api_key
        self.log.info("api key is set")

    def request_key(self, username, password):
        """
        Logins on Jaqpot.

        Parameters
        ----------
        username : username
        password : password

        """
        try:
            au_req = jaqlogin.authenticate_sync(self.base_url, username, password)
            self.api_key = au_req['authToken']
            self.user_id = jwtok.decode_jwt(self.api_key).get('sub')
        except Exception as e:
            self.log.error("Error: " + str(e))

    def request_key_safe(self):
        """
        Logins on Jaqpot by hiding the users password.

        """
        try:
            username = input("Username: ")
            password = getpass.getpass("Password: ")
            au_req = jaqlogin.authenticate_sync(self.base_url, username, password)
            self.api_key = au_req['authToken']
            self.user_id = jwtok.decode_jwt(self.api_key).get('sub')
        except Exception as e:
            self.log.error("Error: " + str(e))

    def my_info(self):
        """
        Prints user's info

        """
        self.log.info(jwtok.decode_jwt(self.api_key))
        return jwtok.decode_jwt(self.api_key)

    def get_algorithms(self, start=None, max=None):
        self.check_key()
        try:
            algos = alapi.get_allgorithms_sync(self.base_url, self.api_key, start, max)
            return algos
        except Exception as e:
            self.log.error("Error:" + str(e))

    def get_algorithms_classes(self, start=None, max=None):
        self.check_key()
        try:
            algos = alapi.get_allgorithms_classes(self.base_url, self.api_key, start, max)
            return algos
        except Exception as e:
            self.log.error("Error:" + str(e))

    def upload_dataset(self, df=None, id=None, title=None, description=None):
        if title is None:
            raise Exception("Please submit title of the dataset")
        if description is None:
            raise Exception("Please submit description of the dataset")
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
        keyi = 0
        for feat in feats:
            feat_info = FeatureInfo()
            f = featapi.create_feature_sync(self.base_url, self.api_key, feat)
            feat_uri = self.base_url + "feature/" + f["_id"]
            feat_map[f["meta"]["titles"][0]] = keyi
            feat_info.uri = feat_uri
            feat_info.name = f["meta"]["titles"][0]
            feat_info.key = keyi
            featutes.append(feat_info.__dict__)
            keyi += 1
        dataset = Dataset()
        meta = MetaInfo()
        meta.creators = [self.user_id]
        if title is not None:
            meta.titles = [title]
        meta.descriptions = [description]
        dataset.meta = meta.__dict__
        dataset.totalRows = df.shape[0]
        dataset.totalColumns = df.shape[1]
        dataset.existence = "UPLOADED"
        data_entry = help.create_data_entry(df, feat_map, self.user_id)
        dataset.dataEntry = data_entry
        dataset.features = featutes
        jsondataset = json.dumps(dataset, cls=JaqpotSerializer)
        dataset_n = data_api.create_dataset_sync(self.base_url, self.api_key, jsondataset, self.log)
        self.log.info("Dataset created with id: " + dataset_n["_id"])
        return dataset_n["_id"]

    def predict(self, df=None, id=None, modelId=None):
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
        model = models_api.get_model(self.base_url, self.api_key, modelId, self.log)
        feats = []
        # for featUri in model['independentFeatures']:
        #     featar = featUri.split("/")
        #     feat = featapi.get_feature(self.base_url, self.api_key, featar[len(featar)-1])
        #     feats.append(feat)
        for featUri in model['additionalInfo']['independentFeatures']:
            print(featUri)
            # featar = featUri.split("/")
            # feat = featapi.get_feature(self.base_url, self.api_key, featar[len(featar)-1])
            # feats.append(feat)
        feat_map = {}
        featutes = []
        keyi = 0
        for feat in feats:
            feat_info = FeatureInfo()
            # f = featapi.create_feature_sync(self.base_url, self.api_key, feat)
            feat_uri = self.base_url + "feature/" + feat["_id"]
            feat_map[feat["meta"]["titles"][0]] = keyi
            feat_info.uri = feat_uri
            feat_info.name = feat["meta"]["titles"][0]
            feat_info.key = keyi
            featutes.append(feat_info.__dict__)
            keyi += 1
        dataset = Dataset()
        meta = MetaInfo()
        meta.creators = [self.user_id]
        dataset.meta = meta.__dict__
        dataset.totalRows = df.shape[0]
        dataset.totalColumns = df.shape[1]
        # dataset.existence = "UPLOADED"
        data_entry = help.create_data_entry(df, feat_map, self.user_id)
        dataset.dataEntry = data_entry
        dataset.features = featutes
        jsondataset = json.dumps(dataset, cls=JaqpotSerializer)
        dataset_n = data_api.create_dataset_sync(self.base_url, self.api_key, jsondataset, self.log)
        datasetId = dataset_n["_id"]
        datasetUri = self.base_url + "dataset/" + datasetId
        task = models_api.predict(self.base_url, self.api_key, dataseturi=datasetUri, modelid=modelId, logger=self.log)
        percentange = 0
        taskid = task['_id']
        while percentange < 100:
            time.sleep(1)
            task = task_api.get_task(self.base_url, self.api_key, taskid)
            try:
                percentange = task['percentageCompleted']
            except KeyError:
                percentange = 0
            self.log.info("completed " + str(percentange))
        predictedDataset = task['resultUri']
        dar = predictedDataset.split("/")
        dataset = data_api.get_dataset(self.base_url, self.api_key, dar[len(dar)-1], self.log)
        df, predicts = ds.decode_predicted(dataset)
        return df, predicts

    # def upload_dataset(self, df=None, id=None, title=None, description=None):
    #     if title in None:
    #         raise Exception("Please submit title of the dataset")
    #     if description is None:
    #         raise Exception("Please submit description of the dataset")
    #     df_titles = list(df)
    #     for t in df_titles:
    #         type_to_c = df[t].dtypes
    #         if type_to_c == 'int64':
    #             # print(type_to_c)
    #             df[t] = df[t].astype(float)
    #     # print(df.dtypes)
    #     df = df.replace(np.nan, '', regex=True)
    #     if type(df).__name__ is not 'DataFrame':
    #         raise Exception("Cannot form a Jaqpot Dataset. Please provide a Dataframe")
    #     if id is not None:
    #         df.set_index(id, inplace=True)
    #         feat_titles = list(df)
    #         feats = []
    #         for f_t in feat_titles:
    #             fe = help.create_feature(f_t, self.user_id)
    #             jsonmi = json.dumps(fe, cls=JaqpotSerializer)
    #             # jsonmi = json.dumps(fe.__dict__)
    #             feats.append(jsonmi)
    #     else:
    #         feat_titles = list(df)
    #         feats = []
    #         for f_t in feat_titles:
    #             fe = help.create_feature(f_t, self.user_id)
    #             # jsonmi = json.dumps(fe.__dict__)
    #             jsonmi = json.dumps(fe, cls=JaqpotSerializer)
    #             feats.append(jsonmi)
    #     feat_map = {}
    #     featutes = []
    #     for feat in feats:
    #         feat_info = FeatureInfo()
    #         f = IOLoop.current().run_sync(lambda: featapi.create_feature_async(self.base_url, self.api_key, feat))
    #         feat_uri = self.base_url + "feature/" + f["_id"]
    #         feat_map[f["meta"]["titles"][0]] = feat_uri
    #         feat_info.uri = feat_uri
    #         feat_info.name = f["meta"]["titles"][0]
    #         featutes.append(feat_info.__dict__)
    #     dataset = Dataset()
    #     meta = MetaInfo()
    #     meta.creators = [self.user_id]
    #     if title is not None:
    #         meta.titles = [title]
    #     meta.descriptions = [description]
    #     dataset.meta = meta.__dict__
    #     dataset.totalRows = df.shape[0]
    #     dataset.totalColumns = df.shape[1]
    #     dataset.existence = "UPLOADED"
    #     data_entry = help.create_data_entry(df, feat_map, self.user_id)
    #     dataset.dataEntry = data_entry
    #     dataset.features = featutes
    #     jsondataset = json.dumps(dataset, cls=JaqpotSerializer)
    #     dataset_n = data_api.create_dataset_sync(self.base_url, self.api_key, jsondataset)
    #     print("Dataset created with id: " + dataset_n["_id"])

    def deploy_linear_model(self, model, X, y, title, description, algorithm, doa=None):
        """
        Deploys sklearn linear_model to Jaqpot.

        Extended description of function.

        Parameters
        ----------
        model : sklearn trained model
            model is a trained model that occurs from the sklearn.linear_model family of algorithms
        X : pandas dataframe
            The dataframe that is used to train the model (X variables).
        y : pandas dataframe
            The dataframe that is used to train the model (y variables).
        title: String
            The title of the model
        description: String
            The description of the model
        algorithm: String
            The algorithm that the model implements
        doa: pandas dataframe
            The dataset used to create the domain of applicability of the model

        Returns
        -------
        string
            The id of the model that uploaded

        """
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
        if doa is None:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Please visit the application to proceed")
                return resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
        else:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Storing Domain of applicability")
                modid = resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
            loop = asyncio.get_event_loop()
            a = loop.create_task(jha.calculate_a(X))
            b = loop.create_task(jha.calculate_doa_matrix(X))
            all_groups = asyncio.gather(a, b)
            results = loop.run_until_complete(all_groups)
            doa = help.create_doa(inv_m=results[1].values.tolist(), a=results[0], modelid=resp['modelId'])
            j = json.dumps(doa, cls=JaqpotSerializer)
            resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
            if resp is 201:
                self.log.info("Stored Domain of applicability. Visit the application to proceed")
                return modid

    def deploy_tree(self, model, X, y, title, description, algorithm,  doa=None):
        """
        Deploys sklearn tree to Jaqpot.

        Extended description of function.

        Parameters
        ----------
        model : sklearn trained model
            model is a trained model that occurs from the sklearn.tree family of algorithms
        X : pandas dataframe
            The dataframe that is used to train the model (X variables).
        y : pandas dataframe
            The dataframe that is used to train the model (y variables).
        title: String
            The title of the model
        description: String
            The description of the model
        algorithm: String
            The algorithm that the model implements
        doa: pandas dataframe
            The dataset used to create the domain of applicability of the model

        Returns
        -------
        string
            The id of the model that uploaded

        """
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
        if doa is None:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Please visit the application to proceed")
                return resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
        else:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Storing Domain of applicability")
                modid = resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
            loop = asyncio.get_event_loop()
            a = loop.create_task(jha.calculate_a(X))
            b = loop.create_task(jha.calculate_doa_matrix(X))
            all_groups = asyncio.gather(a, b)
            results = loop.run_until_complete(all_groups)
            doa = help.create_doa(inv_m=results[1].values.tolist(), a=results[0], modelid=resp['modelId'])
            j = json.dumps(doa, cls=JaqpotSerializer)
            resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
            if resp is 201:
                self.log.info("Stored Domain of applicability. Visit the application to proceed")
                return modid

    def deploy_ensemble(self, model, X, y, title, description, algorithm, doa=None):
        """
        Deploys sklearn ensemble to Jaqpot.

        Extended description of function.

        Parameters
        ----------
        model : sklearn trained model
            model is a trained model that occurs from the sklearn.ensemble family of algorithms
        X : pandas dataframe
            The dataframe that is used to train the model (X variables).
        y : pandas dataframe
            The dataframe that is used to train the model (y variables).
        title: String
            The title of the model
        description: String
            The description of the model
        algorithm: String
            The algorithm that the model implements
        doa: pandas dataframe
            The dataset used to create the domain of applicability of the model

        Returns
        -------
        string
            The id of the model that uploaded

        """
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
        if doa is None:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Please visit the application to proceed")
                return resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
        else:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Storing Domain of applicability")
                modid = resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
            loop = asyncio.get_event_loop()
            a = loop.create_task(jha.calculate_a(X))
            b = loop.create_task(jha.calculate_doa_matrix(X))
            all_groups = asyncio.gather(a, b)
            results = loop.run_until_complete(all_groups)
            doa = help.create_doa(inv_m=results[1].values.tolist(), a=results[0], modelid=resp['modelId'])
            j = json.dumps(doa, cls=JaqpotSerializer)
            resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
            if resp is 201:
                self.log.info("Stored Domain of applicability. Visit the application to proceed")
                return modid

    def deploy_svm(self, model, X, y, title, description, algorithm, doa=None):
        """
        Deploys sklearn svm to Jaqpot.

        Extended description of function.

        Parameters
        ----------
        model : sklearn trained model
            model is a trained model that occurs from the sklearn.svm family of algorithms
        X : pandas dataframe
            The dataframe that is used to train the model (X variables).
        y : pandas dataframe
            The dataframe that is used to train the model (y variables).
        title: String
            The title of the model
        description: String
            The description of the model
        algorithm: String
            The algorithm that the model implements
        doa: pandas dataframe
            The dataset used to create the domain of applicability of the model

        Returns
        -------
        string
            The id of the model that uploaded

        """
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
        if doa is None:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Please visit the application to proceed")
                return resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
        else:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Storing Domain of applicability")
                modid = resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
            loop = asyncio.get_event_loop()
            a = loop.create_task(jha.calculate_a(X))
            b = loop.create_task(jha.calculate_doa_matrix(X))
            all_groups = asyncio.gather(a, b)
            results = loop.run_until_complete(all_groups)
            doa = help.create_doa(inv_m=results[1].values.tolist(), a=results[0], modelid=resp['modelId'])
            j = json.dumps(doa, cls=JaqpotSerializer)
            resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
            if resp is 201:
                self.log.info("Stored Domain of applicability. Visit the application to proceed")
                return modid

    def deploy_cluster(self, model, X, title, description, algorithm,  doa=None):
        """
        Deploys sklearn clustering to Jaqpot.

        Extended description of function.

        Parameters
        ----------
        model : sklearn trained model
            model is a trained model that occurs from the sklearn.svm family of algorithms
        X : pandas dataframe
            The dataframe that is used to train the model (X variables).
        # y : pandas dataframe
        #     The dataframe that is used to train the model (y variables).
        title: String
            The title of the model
        description: String
            The description of the model
        algorithm: String
            The algorithm that the model implements
        doa: pandas dataframe
            The dataset used to create the domain of applicability of the model

        Returns
        -------
        string
            The id of the model that uploaded

        """
        columns = ['Cluster']
        y = pd.DataFrame(columns=columns)
        if isinstance(X, pd.DataFrame) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. X is not one')
        if isinstance(y, pd.DataFrame) is False and isinstance(y, pd.Series) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. Y is not one')
        additionalInfo = {}
        additionalInfo['inputSeries'] = list(X)
        pretrained = help.create_pretrain_req(model, X, y, title, description,
                                              algorithm, "Svm Scikit learn", "scikit-learn-clustering-model",
                                              additionalInfo)
        j = json.dumps(pretrained, cls=JaqpotSerializer)
        if doa is None:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Please visit the application to proceed")
                return resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
        else:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Storing Domain of applicability")
                modid = resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
            loop = asyncio.get_event_loop()
            a = loop.create_task(jha.calculate_a(X))
            b = loop.create_task(jha.calculate_doa_matrix(X))
            all_groups = asyncio.gather(a, b)
            results = loop.run_until_complete(all_groups)
            doa = help.create_doa(inv_m=results[1].values.tolist(), a=results[0], modelid=resp['modelId'])
            j = json.dumps(doa, cls=JaqpotSerializer)
            resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
            if resp is 201:
                self.log.info("Stored Domain of applicability. Visit the application to proceed")
                return modid

    # def deploy_biclustering(self, model, X, y, title, description, algorithm):
    #     """
    #     Deploys sklearn biclustering to Jaqpot.
    #
    #     Extended description of function.
    #
    #     Parameters
    #     ----------
    #     model : sklearn trained model
    #         model is a trained model that occurs from the sklearn.svm family of algorithms
    #     X : pandas dataframe
    #         The dataframe that is used to train the model (X variables).
    #     y : pandas dataframe
    #         The dataframe that is used to train the model (y variables).
    #     title: String
    #         The title of the model
    #     description: String
    #         The description of the model
    #     algorithm: String
    #         The algorithm that the model implements
    #
    #     Returns
    #     -------
    #     string
    #         The id of the model that uploaded
    #
    #     """
    #     if isinstance(X, pd.DataFrame) is False:
    #         raise Exception('Function deploy_glm supports pandas dataframe or series. X is not one')
    #     if isinstance(y, pd.DataFrame) is False and isinstance(y, pd.Series) is False:
    #         raise Exception('Function deploy_glm supports pandas dataframe or series. Y is not one')
    #     additionalInfo = {}
    #     additionalInfo['inputSeries'] = list(X)
    #     pretrained = help.create_pretrain_req(model, X, y, title, description,
    #                                           algorithm, "Svm Scikit learn", "scikit-learn-biclustering-model",
    #                                           additionalInfo)
    #     j = json.dumps(pretrained, cls=JaqpotSerializer)
    #     response = models_api.post_pretrained_model(self.base_url, self.api_key, j)
    #     print("Model with id: " + response['modelId'] + " created. Please visit https://app.jaqpot.org/")

    def deploy_naive_bayess(self, model, X, y, title, description, algorithm,  doa=None):
        """
        Deploys sklearn naive_bayess to Jaqpot.

        Extended description of function.

        Parameters
        ----------
        model : sklearn trained model
            model is a trained model that occurs from the sklearn.svm family of algorithms
        X : pandas dataframe
            The dataframe that is used to train the model (X variables).
        y : pandas dataframe
            The dataframe that is used to train the model (y variables).
        title: String
            The title of the model
        description: String
            The description of the model
        algorithm: String
            The algorithm that the model implements
        doa: pandas dataframe
            The dataset used to create the domain of applicability of the model

        Returns
        -------
        string
            The id of the model that uploaded

        """
        if isinstance(X, pd.DataFrame) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. X is not one')
        if isinstance(y, pd.DataFrame) is False and isinstance(y, pd.Series) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. Y is not one')
        additionalInfo = {}
        additionalInfo['inputSeries'] = list(X)
        pretrained = help.create_pretrain_req(model, X, y, title, description,
                                              algorithm, "Svm Scikit learn", "scikit-learn-naive-bayess-model",
                                              additionalInfo)
        j = json.dumps(pretrained, cls=JaqpotSerializer)
        if doa is None:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Please visit the application to proceed")
                return resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
        else:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Storing Domain of applicability")
                modid = resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
            loop = asyncio.get_event_loop()
            a = loop.create_task(jha.calculate_a(X))
            b = loop.create_task(jha.calculate_doa_matrix(X))
            all_groups = asyncio.gather(a, b)
            results = loop.run_until_complete(all_groups)
            doa = help.create_doa(inv_m=results[1].values.tolist(), a=results[0], modelid=resp['modelId'])
            j = json.dumps(doa, cls=JaqpotSerializer)
            resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
            if resp is 201:
                self.log.info("Stored Domain of applicability. Visit the application to proceed")
                return modid


    def deploy_nearest_neighbors(self, model, X, y, title, description, algorithm, doa=None):
        """
        Deploys sklearn naive_neighbours to Jaqpot.

        Extended description of function.

        Parameters
        ----------
        model : sklearn trained model
            model is a trained model that occurs from the sklearn.svm family of algorithms
        X : pandas dataframe
            The dataframe that is used to train the model (X variables).
        y : pandas dataframe
            The dataframe that is used to train the model (y variables).
        title: String
            The title of the model
        description: String
            The description of the model
        algorithm: String
            The algorithm that the model implements
        doa: pandas dataframe
            The dataset used to create the domain of applicability of the model

        Returns
        -------
        string
            The id of the model that uploaded

        """
        if isinstance(X, pd.DataFrame) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. X is not one')
        if isinstance(y, pd.DataFrame) is False and isinstance(y, pd.Series) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. Y is not one')
        additionalInfo = {}
        additionalInfo['inputSeries'] = list(X)
        pretrained = help.create_pretrain_req(model, X, y, title, description,
                                              algorithm, "Svm Scikit learn", "scikit-learn-nearest-neighbours-model",
                                              additionalInfo)
        j = json.dumps(pretrained, cls=JaqpotSerializer)
        if doa is None:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Please visit the application to proceed")
                return resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
        else:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Storing Domain of applicability")
                modid = resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
            loop = asyncio.get_event_loop()
            a = loop.create_task(jha.calculate_a(X))
            b = loop.create_task(jha.calculate_doa_matrix(X))
            all_groups = asyncio.gather(a, b)
            results = loop.run_until_complete(all_groups)
            doa = help.create_doa(inv_m=results[1].values.tolist(), a=results[0], modelid=resp['modelId'])
            j = json.dumps(doa, cls=JaqpotSerializer)
            resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
            if resp is 201:
                self.log.info("Stored Domain of applicability. Visit the application to proceed")
                return modid

    def deploy_neural_network(self, model, X, y, title, description, algorithm, doa=None):
        """
        Deploys sklearn neural_network to Jaqpot.

        Extended description of function.

        Parameters
        ----------
        model : sklearn trained model
            model is a trained model that occurs from the sklearn.svm family of algorithms
        X : pandas dataframe
            The dataframe that is used to train the model (X variables).
        y : pandas dataframe
            The dataframe that is used to train the model (y variables).
        title: String
            The title of the model
        description: String
            The description of the model
        algorithm: String
            The algorithm that the model implements
        doa: pandas dataframe
            The dataset used to create the domain of applicability of the model

        Returns
        -------
        string
            The id of the model that uploaded

        """
        if isinstance(X, pd.DataFrame) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. X is not one')
        if isinstance(y, pd.DataFrame) is False and isinstance(y, pd.Series) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. Y is not one')
        additionalInfo = {}
        additionalInfo['inputSeries'] = list(X)
        pretrained = help.create_pretrain_req(model, X, y, title, description,
                                              algorithm, "Svm Scikit learn", "scikit-learn-neural-network-model",
                                              additionalInfo)
        j = json.dumps(pretrained, cls=JaqpotSerializer)
        if doa is None:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Please visit the application to proceed")
                return resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
        else:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Storing Domain of applicability")
                modid = resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
            loop = asyncio.get_event_loop()
            a = loop.create_task(jha.calculate_a(X))
            b = loop.create_task(jha.calculate_doa_matrix(X))
            all_groups = asyncio.gather(a, b)
            results = loop.run_until_complete(all_groups)
            doa = help.create_doa(inv_m=results[1].values.tolist(), a=results[0], modelid=resp['modelId'])
            j = json.dumps(doa, cls=JaqpotSerializer)
            resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
            if resp is 201:
                self.log.info("Stored Domain of applicability. Visit the application to proceed")
                return modid

    def deploy_pipeline(self, pipeline, X, y, title, description, algorithm, doa=None):
        """
        Deploys sklearn pipeline to Jaqpot.

        Extended description of function.

        Parameters
        ----------
        pipeline : sklearn pipeline
            model is a trained model that occurs from the sklearn.svm family of algorithms
        X : pandas dataframe
            The dataframe that is used to train the model (X variables).
        y : pandas dataframe
            The dataframe that is used to train the model (y variables).
        title: String
            The title of the model
        description: String
            The description of the model
        algorithm: String
            The algorithm that the model implements
        doa: pandas dataframe
            The dataset used to create the domain of applicability of the model

        Returns
        -------
        string
            The id of the pipeline that uploaded

        """
        if isinstance(X, pd.DataFrame) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. X is not one')
        if isinstance(y, pd.DataFrame) is False and isinstance(y, pd.Series) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. Y is not one')
        additionalInfo = {}
        additionalInfo['inputSeries'] = list(X)
        pretrained = help.create_pretrain_req(pipeline, X, y, title, description,
                                              algorithm, "Svm Scikit learn", "scikit-learn-pipeline",
                                              additionalInfo)
        j = json.dumps(pretrained, cls=JaqpotSerializer)
        if doa is None:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Please visit the application to proceed")
                return resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
        else:
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                self.log.info("Model with id: " + resp['modelId'] + " created. Storing Domain of applicability")
                modid = resp['modelId']
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return
            loop = asyncio.get_event_loop()
            a = loop.create_task(jha.calculate_a(X))
            b = loop.create_task(jha.calculate_doa_matrix(X))
            all_groups = asyncio.gather(a, b)
            results = loop.run_until_complete(all_groups)
            doa = help.create_doa(inv_m=results[1].values.tolist(), a=results[0], modelid=resp['modelId'])
            j = json.dumps(doa, cls=JaqpotSerializer)
            resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
            if resp is 201:
                self.log.info("Stored Domain of applicability. Visit the application to proceed")
                return modid

    def api_key(self):
        return self.api_key

    def check_key(self):
        if self.api_key is None:
            raise Exception("Api key not present!")
