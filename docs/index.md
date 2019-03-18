# Welcome to jaqpotpy documentation

### About

jaqpotpy is a library that lets you deploy and share seemingless easy machine learning and statistal models over the web
It is created by the [Unit of Process Control and Informatics in the School of Chemical Engineering](https://www.chemeng.ntua.gr/labs/control_lab/) at the National Technical University of Athens.

### Contact

pantelispanka@gmail.com


#### The application can be found at https://app.jaqpot.org



## installation

Jaqpotpy can be installed as a pypi package.

* `pip install jaqpotpy` 


## Usage and initialization

### Import Jaqpot

* `from jaqpotpy import Jaqpot`


### Initialize Jaqpotpy on the services that jaqpot lives.


* `jaqpot = Jaqpot("https://api.jaqpot.org/jaqpot/services/")`


### Let jaqpot know who you are

Login and have access on the jaqpot services

In order to do so you can use the functions:

* `jaqpot.login('username', 'password')`

Will login and set the api key that is needed.

* `jaqpot.request_key('username', 'password')`

Same as above you request the key and set it on jaqpot

* `jaqpot.request_key_safe()`

Will ask the user for the username and password by hidding the password if 
jaqpot is used through a jupiter notebook etc

### Set Key without login

Some users may have logged in through google or github. At the account page 
a user can find an api key that can be used in order to have access on the services.
These keys have short life and should be updated on each login.

* `jaqpot.set_api_key("api_key")`


## Deploy your models!

**Jaqpot and jaqpotpy are now on betta version!!!**

**We cannot guarantee the existence of the data**

**Not everything is tested thoroughly and thus things may not work as well!**

**Contact and let us know for any mistakes**

## deploy_linear_model()

* `jaqpot.deploy_linear_model()`

Let's you deploy linear models that are created from scikit-learn

Bellow there is a list for the produced models that can be deployed with this function


- linear_model.ARDRegression()
- linear_model.BayesianRidge()	
- linear_model.ElasticNet()
- linear_model.ElasticNetCV()	
- linear_model.HuberRegressor()
- linear_model.Lars()	
- linear_model.LarsCV()	
- linear_model.Lasso()	
- linear_model.LassoCV()
- linear_model.LassoLars()
- linear_model.LassoLarsCV()	
- linear_model.LassoLarsIC()	
- linear_model.LinearRegression()	
- linear_model.LogisticRegression()
- linear_model.LogisticRegressionCV()	
- linear_model.MultiTaskLasso()	
- linear_model.MultiTaskElasticNet()
- linear_model.MultiTaskLassoCV()	
- linear_model.MultiTaskElasticNetCV()
- linear_model.OrthogonalMatchingPursuit()	
- linear_model.OrthogonalMatchingPursuitCV()	
- linear_model.PassiveAggressiveClassifier()	
- linear_model.PassiveAggressiveRegressor()	
- linear_model.Perceptron()
- linear_model.RANSACRegressor()
- linear_model.Ridge()
- linear_model.RidgeClassifier()
- linear_model.RidgeClassifierCV()
- linear_model.RidgeCV()
- linear_model.SGDClassifier()
- linear_model.SGDRegressor()
- linear_model.TheilSenRegressor()	
- linear_model.enet_path()
- linear_model.lars_path()	
- linear_model.lasso_path()
- linear_model.logistic_regression_path()
- linear_model.orthogonal_mp()
- linear_model.orthogonal_mp_gram()	
- linear_model.ridge_regression()


**deploy_linear_model() parameters are:**

- model : sklearn trained model
model is a trained model that occurs from the sklearn.linear_model family of algorithms
- X : pandas dataframe
The dataframe that is used to train the model (X variables).
- y : pandas dataframe
The dataframe that is used to train the model (y variables).
- title: String
The title of the model
- description: String
The description of the model
- algorithm: String
The algorithm that the model implements
string


The id of the model is returned. The model can be found on the home page of the user for editing / sharing / execution (create predictions)


### Example usage

````
from jaqpotpy import Jaqpot
import pandas as pd
from sklearn import linear_model



df2 = pd.read_csv('/path/train.csv')
X2 = df2[['Pclass',  'SibSp', 'Parch', 'Fare']]
y2 = df2['Survived']

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X2, y2)


jaqpot.deploy_linear_model(clf, X2, y2, title="Sklearn 2", description="Logistic regression model from python for the titanic dataset",
                  algorithm="logistic regression")
````


On the above example a linear  model (in our case a logistic regression) is created and deployed on jaqpot.

The dataset is read with pandas and we created the X and y dataframes on which we trained the algorithm and created the model

For this example we used the Fame [Titanic dataset](https://www.kaggle.com/c/titanic). Any dataset could be used


**The models should be trained with pandas dataframe!**


## deploy_cluster()


Let's you deploy cluster models that are created from scikit-learn

- cluster.AffinityPropagation()
- cluster.AgglomerativeClustering()
- cluster.Birch()	
- cluster.DBSCAN()	
- cluster.FeatureAgglomeration()	
- cluster.KMeans()
- cluster.MiniBatchKMeans()
- cluster.MeanShift()
- cluster.SpectralClustering()


**jaqpot.deploy_deploy_cluster() parameters are:**

- model : sklearn trained model
model is a trained model that occurs from the sklearn.linear_model family of algorithms
- X : pandas dataframe
The dataframe that is used to train the model (X variables).
- title: String
The title of the model
- description: String
The description of the model
- algorithm: String
The algorithm that the model implements
string

The id of the model is returned. The model can be found on the home page of the user for editing / sharing / execution (create predictions)


## deploy_ensemble()

Let's you deploy cluster models that are created from scikit-learn

Algorithms:

- ensemble.AdaBoostClassifier()
- ensemble.AdaBoostRegressor()
- ensemble.BaggingClassifier()	
- ensemble.BaggingRegressor()
- ensemble.ExtraTreesClassifier()
- ensemble.ExtraTreesRegressor()
- ensemble.GradientBoostingClassifier()
- ensemble.GradientBoostingRegressor()
- ensemble.IsolationForest()
- ensemble.RandomForestClassifier()
- ensemble.RandomForestRegressor()
- ensemble.RandomTreesEmbedding()	
- ensemble.VotingClassifier()	


**jaqpot.deploy_ensemble() parameters are:**

- model : sklearn trained model
model is a trained model that occurs from the sklearn.linear_model family of algorithms
- X : pandas dataframe
The dataframe that is used to train the model (X variables).
- y : pandas dataframe
The dataframe that is used to train the model (y variables).
- title: String
The title of the model
- description: String
The description of the model
- algorithm: String
The algorithm that the model implements
string


The id of the model is returned. The model can be found on the home page of the user for editing / sharing / execution (create predictions)


## deploy_naive_bayess()

Let's you deploy naive_bayess models that are created from scikit-learn

Algorithms:

naive_bayes.BernoulliNB()	
naive_bayes.GaussianNB()
naive_bayes.MultinomialNB()	
naive_bayes.ComplementNB()	


**jaqpot.deploy_naive_bayess() parameters are:**

- model : sklearn trained model
model is a trained model that occurs from the sklearn.linear_model family of algorithms
- X : pandas dataframe
The dataframe that is used to train the model (X variables).
- y : pandas dataframe
The dataframe that is used to train the model (y variables).
- title: String
The title of the model
- description: String
The description of the model
- algorithm: String
The algorithm that the model implements
string


The id of the model is returned. The model can be found on the home page of the user for editing / sharing / execution (create predictions)


## nearest_neighbors()

Let's you deploy nearest_neighbors models that are created from scikit-learn

Algorithms:

neighbors.KNeighborsClassifier()
neighbors.KNeighborsRegressor()	
neighbors.LocalOutlierFactor()
neighbors.RadiusNeighborsClassifier()
neighbors.RadiusNeighborsRegressor()	
neighbors.NearestCentroid()
neighbors.NearestNeighbors()
neighbors.kneighbors_graph()
neighbors.radius_neighbors_graph()


**jaqpot.deploy_nearest_neighbors() parameters are:**

- model : sklearn trained model
model is a trained model that occurs from the sklearn.linear_model family of algorithms
- X : pandas dataframe
The dataframe that is used to train the model (X variables).
- y : pandas dataframe
The dataframe that is used to train the model (y variables).
- title: String
The title of the model
- description: String
The description of the model
- algorithm: String
The algorithm that the model implements
string


**if y is empty generate an empty dataframe with the title of the predicted feature**

The id of the model is returned. The model can be found on the home page of the user for editing / sharing / execution (create predictions)


## deploy_neural_network()

Let's you deploy neural_network models that are created from scikit-learn

Algorithms:


- neural_network.BernoulliRBM()	
- neural_network.MLPClassifier()	
- neural_network.MLPRegressor()	


**jaqpot.deploy_neural_network() parameters are:**

- model : sklearn trained model
model is a trained model that occurs from the sklearn.linear_model family of algorithms
- X : pandas dataframe
The dataframe that is used to train the model (X variables).
- y : pandas dataframe
The dataframe that is used to train the model (y variables).
- title: String
The title of the model
- description: String
The description of the model
- algorithm: String
The algorithm that the model implements
string


The id of the model is returned. The model can be found on the home page of the user for editing / sharing / execution (create predictions)



## deploy_svm()

Let's you deploy svm models that are created from scikit-learn

Algorithms:


- svm.LinearSVC()	
- svm.LinearSVR()	
- svm.NuSVC()	
- svm.NuSVR()
- svm.OneClassSVM()
- svm.SVC()	
- svm.SVR()	
- svm.l1_min_c()	


**jaqpot.deploy_svm() parameters are:**

- model : sklearn trained model
model is a trained model that occurs from the sklearn.linear_model family of algorithms
- X : pandas dataframe
The dataframe that is used to train the model (X variables).
- y : pandas dataframe
The dataframe that is used to train the model (y variables).
- title: String
The title of the model
- description: String
The description of the model
- algorithm: String
The algorithm that the model implements
string

**if y is empty generate an empty dataframe with the title of the predicted feature**


The id of the model is returned. The model can be found on the home page of the user for editing / sharing / execution (create predictions)


## deploy_tree()

Let's you deploy tree models that are created from scikit-learn

Algorithms:


- tree.DecisionTreeClassifier()	
- tree.DecisionTreeRegressor()
- tree.ExtraTreeClassifier()	
- tree.ExtraTreeRegressor()	


**jaqpot.deploy_tree() parameters are:**

- model : sklearn trained model
model is a trained model that occurs from the sklearn.linear_model family of algorithms
- X : pandas dataframe
The dataframe that is used to train the model (X variables).
- y : pandas dataframe
The dataframe that is used to train the model (y variables).
- title: String
The title of the model
- description: String
The description of the model
- algorithm: String
The algorithm that the model implements
string


The id of the model is returned. The model can be found on the home page of the user for editing / sharing / execution (create predictions)



## deploy_pipeline()

Let's you deploy pipelined models that are created from scikit-learn
	


**jaqpot.deploy_pipeline() parameters are:**

- pipeline : sklearn pipeline
model is a trained model that occurs from the sklearn.linear_model family of algorithms
- X : pandas dataframe
The dataframe that is used to train the model (X variables).
- y : pandas dataframe
The dataframe that is used to train the model (y variables).
- title: String
The title of the model
- description: String
The description of the model
- algorithm: String
The algorithm that the model implements
string


The id of the model / pipeline is returned. The model can be found on the home page of the user for editing / sharing / execution (create predictions)





## Example usage


````
from jaqpotpy import Jaqpot
import pandas as pd
from sklearn import linear_model





df2 = pd.read_csv('/path/train.csv')
X2 = df2[['Pclass',  'SibSp', 'Parch', 'Fare']]
y2 = df2['Survived']

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X2, y2)


jaqpot.deploy_linear_model(clf, X2, y2, title="Sklearn 2", description="Logistic regression model from python for the titanic dataset",
                  algorithm="logistic regression")
````


On the above example a linear  model (in our case a logistic regression) is created and deployed on jaqpot.

The dataset is read with pandas and we created the X and y dataframes on which we trained the algorithm and created the model

For this example we used the Fame [Titanic dataset](https://www.kaggle.com/c/titanic). Any dataset could be used


**The models should be trained with pandas dataframe!**
