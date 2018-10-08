from jaqpotpy import Jaqpot
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from base64 import b64encode, b64decode
ENCODING = 'utf-8'

# jaqp = jap.Jaqpot("https://api.jaqpot.org/jaqpot/services/")

jaqpot = Jaqpot("http://localhost:8080/jaqpot/services/")
jaqpot.request_key("pantelispanka", "kapan1")
# jaqp.request_key_safe()

# jaqpot.my_info()
# algos = jaqpot.get_algorithms()


# print(algos)

# algos_classes = jaqp.get_algorithms_classes()
# for algo in algos_classes:
#     print(algo.meta)

df = pd.read_csv('/Users/pantelispanka/Desktop/gdp-countries.csv')


# jaqpot.upload_dataset(df=df, id='country')

lm = LinearRegression()

y = df['GDP']
X = df[['LFG', 'EQP', 'NEQ', 'GAP']]

model = lm.fit(X=X, y=y)

p_mod = pickle.dumps(model)
raw_model = b64encode(p_mod)
raw_model_string = raw_model.decode(ENCODING)


# print(list(X))
# print(y.name)
# print(model.coef_)

pred = model.predict(X)

p_raw = b64decode(raw_model_string)
raw_ = pickle.loads(p_raw)
pred_r = raw_.predict(X)


jaqpot.deploy_glm(model, X, y, title="Sklearn linear", description="First pretrained from python",
                  algorithm="linear_model")

# print(pred_r)
# print(pred)

res_plot = sns.residplot(y, pred, lowess=True, color="g")
# plt.show()

# plt.scatter(X, y,  color='black')
# plt.plot(X, model.predict(X), color='blue', linewidth=1)
# plt.xticks(())
# plt.yticks(())
# plt.show()


# print(len(df['country'].unique().tolist()), len(df['GDP'].unique().tolist()), len(df['LFG'].unique().tolist()),
#       len(df['EQP'].unique().tolist()))
# print(df.get_values())
