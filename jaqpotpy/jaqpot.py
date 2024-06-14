import jaqpotpy.api.login as jaqlogin
import jaqpotpy.api.algorithms_api as alapi
import jaqpotpy.api.dataset_api as data_api
import jaqpotpy.api.models_api as models_api
import jaqpotpy.api.chempot_api as chempot_api
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
import jaqpotpy.doa.doa as jha
from sys import getsizeof
from tqdm import tqdm


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
            self.base_url = 'https://api.jaqpot.org/jaqpot/services/'
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
            # self.user_id = jwtok.decode_jwt(self.api_key).get('sub')
        except Exception as e:
            self.log.error("Could not login to jaqpot")
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
            self.log.info("api key is set")
            # self.user_id = jwtok.decode_jwt(self.api_key).get('sub')
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
            self.log.info("api key is set")
            # self.user_id = jwtok.decode_jwt(self.api_key).get('sub')
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
        if type(df).__name__ != 'DataFrame':
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

    def chempot_predict(self, model, smiles, descriptors = "mordred", doa = False):
        """
        Makes a prediction from a Chempot model

        Parameters
        ----------
        model : string
            model is the id of the chempot model
        smiles : string
            The input SMILES.
        descriptors : string
            The descriptors of the model.
        doa : boolean
            Whether the Domain of Applicability will be taken into account
        
        Returns
        -------
        tuple
            The predicted dataset (first item) and the predicted feature (second item)
        """
        task = chempot_api.predict(self.base_url, self.api_key, model, smiles, descriptors, doa, self.log)
        
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


    def predict(self, df=None, modelId=None):
        df_titles = list(df)
        for t in df_titles:
            type_to_c = df[t].dtypes
            # if type_to_c == 'int64':
                # print(type_to_c)
                # df[t] = df[t].astype(float)
        # print(df.dtypes)
        df = df.replace(np.nan, '', regex=True)
        if type(df).__name__ != 'DataFrame':
            raise Exception("Cannot form a Jaqpot Dataset. Please provide a Dataframe")
        model = models_api.get_model(self.base_url, self.api_key, modelId, self.log)
        # feats = []
        # for featUri in model['independentFeatures']:
        #     featar = featUri.split("/")
        #     feat = featapi.get_feature(self.base_url, self.api_key, featar[len(featar)-1])
        #     feats.append(feat)
        # for featUri in model['additionalInfo']['independentFeatures']:
        #     print(featUri)
            # featar = featUri.split("/")
            # feat = featapi.get_feature(self.base_url, self.api_key, featar[len(featar)-1])
            # feats.append(feat)
        feat_map = {}
        featutes = []
        keyi = 0

        for key,value in model['additionalInfo']['independentFeatures'].items():
            feat_info = FeatureInfo()
            # f = featapi.create_feature_sync(self.base_url, self.api_key, feat)
            feat_uri = key
            feat_map[value] = keyi
            feat_info.uri = feat_uri
            feat_info.name = value
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

    def deploy_sklearn(self, model, X, y, title, description, model_meta=False, doa=None):
        """
        Deploys sklearn model or pipeline to Jaqpot.

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
        doa: pandas dataframe
            The dataset used to create the domain of applicability of the model

        Returns
        -------
        string
            The id of the model that uploaded
        """

        algorithm = type(model).__name__
        sk_version = model.__getstate__()['_sklearn_version']

        splited = sk_version.split(".")
        if int(splited[0]) == 1 and int(splited[1]) == 5:
            runtime = "jaqpotpy"
        else:
            raise Exception('Only Scikit-Learn 1.5.0 is supported. Please update your scikit-learn version to 1.5.0')

        if isinstance(X, pd.DataFrame) is False and isinstance(X, pd.Series) is False:
            raise Exception('Function deploy_sklearn supports pandas dataframe or series. X is not one')
        if isinstance(y, pd.DataFrame) is False and isinstance(y, pd.Series) is False:
            raise Exception('Function deploy_sklearn supports pandas dataframe or series. Y is not one')

        additionalInfo = {}
        if model_meta is True:
            additionalInfo['meta'] = model.__getstate__()
        else:
            additionalInfo['meta'] = {}
            additionalInfo['meta']['_sklearn_version'] = sk_version
        additionalInfo['meta']['algorithm'] = algorithm
        if isinstance(X, pd.DataFrame):
            additionalInfo['inputSeries'] = list(X)
        if isinstance(X, pd.Series):
            additionalInfo['inputSeries'] = X.name
        pretrained = help.create_pretrain_req(model, X, y, title, description,
                                              algorithm, "Scikit learn model or pipeline", runtime,
                                              additionalInfo)
        size = getsizeof(pretrained.rawModel[0])
        if size < 2000000:
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
                # loop = asyncio.get_event_loop()
                # a = loop.create_task(jha.calculate_a(X))
                # b = loop.create_task(jha.calculate_doa_matrix(X))
                a = jha.calculate_a(X)
                b = jha.calculate_doa_matrix(X)
                # all_groups = asyncio.gather(a, b)
                # results = loop.run_until_complete(all_groups)
                doa = help.create_doa(inv_m=b.values.tolist(), a=a, modelid=resp['modelId'])
                j = json.dumps(doa, cls=JaqpotSerializer)
                resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
                if resp == 201:
                    self.log.info("Stored Domain of applicability. Visit the application to proceed")
                    return modid
        else:
            rawModel = pretrained.rawModel[0]
            rawModel = str(rawModel)
            n = 1000000
            chunks = [rawModel[i:i + n] for i in range(0, len(rawModel), n)]
            del pretrained.rawModel
            j = json.dumps(pretrained, cls=JaqpotSerializer)
            if doa is None:
                response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
                if response.status_code < 300:
                    resp = response.json()
                    model_id = resp['modelId']
                    parts = []
                    i = 0
                    for c in chunks:
                        part = {}
                        part['modelId'] = model_id
                        part['partNumber'] = i
                        part['totalParts'] = len(chunks)
                        part['part'] = c
                        parts.append(part)
                        i += 1
                    for j in tqdm(range(len(chunks))):
                        part = parts[j]
                        json_r = json.dumps(part, cls=JaqpotSerializer)
                        resp = models_api.post_model_part(self.base_url, self.api_key, model_id, json_r, self.log)
                        # pass
                    # resp = response.json()
                    self.log.info("Model with id: " + model_id + " created. Please visit the application to proceed")
                    return model_id
                else:
                    resp = response.json()
                    self.log.error("Some error occured: " + resp['message'])
                    return
            else:
                response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
                resp = response.json()
                if response.status_code < 300:
                    resp = response.json()
                    model_id = resp['modelId']
                    self.log.info("Model with id: " + resp['modelId'] + " created. Storing Domain of applicability")
                    modid = resp['modelId']
                else:
                    resp = response.json()
                    self.log.error("Some error occured: " + resp['message'])
                    return
                a = jha.calculate_a(X)
                b = jha.calculate_doa_matrix(X)
                doa = help.create_doa(inv_m=b.values.tolist(), a=a, modelid=resp['modelId'])
                j = json.dumps(doa, cls=JaqpotSerializer)
                resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
                if resp == 201:
                    self.log.info("Stored Domain of applicability. Visit the application to proceed")
                    return modid
            # print("Not supported")

    def deploy_jaqpotpy_molecular_model(self, model, description: str, title: str = None):
        if model.model_title != None:
            title = model.model_title
        pretrained = help.create_molecular_req(model, title=title, description=description, type="MolecularModel")
        size = getsizeof(pretrained.rawModel[0])
        if size < 2000000:
            j = json.dumps(pretrained, cls=JaqpotSerializer)
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
            rawModel = pretrained.rawModel[0]
            rawModel = str(rawModel)
            n = 2000000
            chunks = [rawModel[i:i + n] for i in range(0, len(rawModel), n)]
            del pretrained.rawModel
            j = json.dumps(pretrained, cls=JaqpotSerializer)
            response = models_api.post_pretrained_model(self.base_url, self.api_key, j, self.log)
            if response.status_code < 300:
                resp = response.json()
                model_id = resp['modelId']
                parts = []
                i = 0
                for c in chunks:
                    part = {}
                    part['modelId'] = model_id
                    part['partNumber'] = i
                    part['totalParts'] = len(chunks)
                    part['part'] = c
                    parts.append(part)
                    i += 1
                for j in tqdm(range(len(chunks))):
                    part = parts[j]
                    json_r = json.dumps(part, cls=JaqpotSerializer)
                    resp = models_api.post_model_part(self.base_url, self.api_key, model_id, json_r, self.log)
                    # pass
                # resp = response.json()
                self.log.info("Model with id: " + model_id + " created. Please visit the application to proceed")
                return model_id
            else:
                resp = response.json()
                self.log.error("Some error occured: " + resp['message'])
                return

    def deploy_sklearn_unsupervised(self, model, X, title, description, model_meta=False, doa=None):
        """
        Deploys sklearn model or pipeline to Jaqpot.

        Extended description of function.

        Parameters
        ----------
        model : sklearn trained model
            model is a trained model that occurs from the sklearn.linear_model family of algorithms
        X : pandas dataframe
            The dataframe that is used to train the model (X variables).
        title: String
            The title of the model
        description: String
            The description of the model
        doa: pandas dataframe
            The dataset used to create the domain of applicability of the model

        Returns
        -------
        string
            The id of the model that uploaded
        """

        algorithm = type(model).__name__
        sk_version = model.__getstate__()['_sklearn_version']

        splited = sk_version.split(".")
        if int(splited[1]) < 21 and int(splited[0]) == 0:
            runtime = "scikit-learn-legacy"
        if int(splited[1]) == 22:
            runtime = "scikit-learn-0-22"
        if int(splited[1]) == 23:
            runtime = "scikit-learn-0-23"
        if int(splited[1]) == 24:
            runtime = "scikit-learn-0-24"
        if int(splited[1]) > 24:
            runtime = "scikit-learn-0-24"
        if int(splited[0]) == 1 and int(splited[1]) == 0:
            runtime = "scikit-learn-1-0"
        if int(splited[0]) == 1 and int(splited[1]) == 1:
            runtime = "scikit-learn-1-1"

        if isinstance(X, pd.DataFrame) is False:
            raise Exception('Function deploy_glm supports pandas dataframe or series. X is not one')
        # if isinstance(y, pd.DataFrame) is False and isinstance(y, pd.Series) is False:
        #     raise Exception('Function deploy_glm supports pandas dataframe or series. Y is not one')

        y = pd.Series(data=None, index=['y'])

        additionalInfo = {}
        if model_meta is True:
            additionalInfo['meta'] = model.__getstate__()
        else:
            additionalInfo['meta'] = {}
            additionalInfo['meta']['_sklearn_version'] = sk_version
        additionalInfo['meta']['algorithm'] = algorithm
        if isinstance(X, pd.DataFrame):
            additionalInfo['inputSeries'] = list(X)
        if isinstance(X, pd.Series):
            additionalInfo['inputSeries'] = X.name
        pretrained = help.create_pretrain_req(model, X, y, title, description,
                                              algorithm, "Scikit learn model or pipeline", runtime,
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
            # loop = asyncio.get_event_loop()
            # a = loop.create_task(jha.calculate_a(X))
            # b = loop.create_task(jha.calculate_doa_matrix(X))
            # all_groups = asyncio.gather(a, b)
            # results = loop.run_until_complete(all_groups)
            a = jha.calculate_a(X)
            b = jha.calculate_doa_matrix(X)
            doa = help.create_doa(inv_m=b.values.tolist(), a=a, modelid=resp['modelId'])
            j = json.dumps(doa, cls=JaqpotSerializer)
            resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
            if resp == 201:
                self.log.info("Stored Domain of applicability. Visit the application to proceed")
                return modid

    def api_key(self):
        return self.api_key

    def check_key(self):
        if self.api_key is None:
            raise Exception("Api key not present!")

    def deploy_XGBoost(self, model, X, y, title, description, algorithm, doa=None):
        """
        Deploys XGBoost model to Jaqpot.

        Extended description of function.

        Parameters
        ----------
        model : XGBoost trained model
            model is a trained model that occurs from the XGBoost family of algorithms
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
        # coef_flatten = model.coef_.flatten()
        # intercept_flatten = model.intercept_.flatten()
        # coef_df = pd.DataFrame(list(zip(list(X), coef_flatten)), columns=['Features', 'coeff'])
        # intercept = pd.DataFrame([('(intercept)', intercept_flatten[0])], columns=['Features', 'coeff'])
        # coefs_all = intercept.append(coef_df, ignore_index=True)
        # coefs = {}
        additionalInfo = {}
        # for key in coefs_all.values:
        #     coefs[key[0]] = key[1]
        # additionalInfo['coefficients'] = coefs
        additionalInfo['inputSeries'] = list(X)
        pretrained = help.create_pretrain_req(model, X, y, title, description,
                                              algorithm, "XGBoost model", "XGBoost-model",
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
            # loop = asyncio.get_event_loop()
            # a = loop.create_task(jha.calculate_a(X))
            # b = loop.create_task(jha.calculate_doa_matrix(X))

            a = jha.calculate_a(X)
            b = jha.calculate_doa_matrix(X)
            # all_groups = asyncio.gather(a, b)
            # results = loop.run_until_complete(all_groups)
            doa = help.create_doa(inv_m=b.values.tolist(), a=a, modelid=resp['modelId'])
            j = json.dumps(doa, cls=JaqpotSerializer)
            resp = doa_api.post_models_doa(self.base_url, self.api_key, j, self.log)
            if resp == 201:
                self.log.info("Stored Domain of applicability. Visit the application to proceed")
                return modid

    def get_model_by_id(self, model):
        """
        Retrieves user's model by ID.

        Parameters
        ----------
        model : str
            The model's ID.
        
        
        Returns
        -------
        Object
            The particular model.

        """
        return models_api.get_model(self.base_url, self.api_key, model, self.log)

    def get_raw_model_by_id(self, model):
        """
        Retrieves raw model by ID.

        Parameters
        ----------
        model : str
            The model's ID.

        Returns
        Returns
        -------
        Object
            The particular model and the raw model.
        """

        # raw_model = models_api.get_raw_model(self.base_url, self.api_key, model, self.log)

        validating = jaqlogin.validate_api_key(self.base_url, self.api_key)
        if validating is not True:
            self.log.error(validating)
        else:
            return models_api.get_raw_model(self.base_url, self.api_key, model, self.log)

    def get_feature_by_id(self, feature):
        """
        Retrieves a Jaqpot feature.

        Parameters
        ----------
        feature : str
            The feature's ID.
        
        
        Returns
        -------
        Object
            The particular feature.

        """
        return featapi.get_feature(self.base_url, self.api_key, feature)

    def get_my_models(self, minimum, maximum):
        """
        Retrieves user's models.

        Parameters
        ----------
        minimum : int
            The index of the first model.
        maximum : int
            The index of the last model.
        
        Returns
        -------
        Object
            The models of the user.

        """
        return models_api.get_my_models(self.base_url, self.api_key, minimum, maximum, self.log)

    
    def get_orgs_models(self, organization, minimum, maximum):
        """
        Retrieves organization's models.

        Parameters
        ----------
        organization: str
            The organization's ID.
        minimum : int
            The index of the first model.
        maximum : int
            The index of the last model.
        
        Returns
        -------
        Object
            The models of the organization.

        """
        return models_api.get_orgs_models(self.base_url, self.api_key, organization, minimum, maximum, self.log)


    def get_models_by_tag(self, tag, minimum, maximum):
        """
        Retrieves models with a particular tag.

        Parameters
        ----------
        tag: str
            The model's tag.
        minimum : int
            The index of the first model.
        maximum : int
            The index of the last model.
        
        Returns
        -------
        Object
            The models of the organization.

        """
        return models_api.get_models_by_tag(self.base_url, self.api_key, tag, minimum, maximum, self.log)

    def get_models_by_tag_and_org(self, organization, tag, minimum, maximum):
        """
        Retrieves models of an organization with a particular tag.

        Parameters
        ----------
        organization: str:
            The organization's ID.
        tag: str
            The model's tag.
        minimum : int
            The index of the first model.
        maximum : int
            The index of the last model.
        
        Returns
        -------
        Object
            The models of the organization.

        """
        return models_api.get_models_by_tag(self.base_url, self.api_key, organization, tag, minimum, maximum, self.log)


    def get_dataset(self, dataset):
        """
        Retrieves a dataset.

        Parameters
        ----------
        dataset: str
            The dataset's ID.
        
        Returns
        -------
        pd.DataFrame()
            DataFrame of the dataset.

        Note
        ----
        Mazimum rows retrieved: 1000.
        """

        dataRetrieved = data_api.get_dataset(self.base_url, self.api_key, dataset, self.log)

        feats = ["" for i in range(len(dataRetrieved['features']))]
        
        for item in dataRetrieved['features']:
            feats[int(item['key'])] = item['name']

        indices = [item['entryId']['name'] for item in dataRetrieved['dataEntry']]

        data = [[item['values'][str(feats.index(col))] for col in feats] for item in dataRetrieved['dataEntry']]

        return pd.DataFrame(data,index=indices, columns=feats) 

    def get_doa(self,model):
        """
        Retrieves model's domain of applicability.

        Parameters
        ----------
        model: str
            The model's ID.
        
        Returns
        -------
        Object
            The model's domain of applicability.

        """
        return doa_api.get_models_doa(self.base_url, self.api_key, model, self.log)
