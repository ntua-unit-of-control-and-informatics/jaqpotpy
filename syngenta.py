from jaqpotpy import Jaqpot
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv('/Users/pantelispanka/Downloads/LTKB_mordred.csv')

print(list(df))

X = df[['SpMax_Dzare', 'SpDiam_Dzare', 'SpAD_Dzare', 'SpMAD_Dzare', 'LogEE_Dzare', 'SM1_Dzare', 'VE1_Dzare', 'VE2_Dzare', 'VE3_Dzare', 'VR1_Dzare', 'VR2_Dzare', 'VR3_Dzare', 'SpAbs_Dzp', 'SpMax_Dzp', 'SpDiam_Dzp', 'SpAD_Dzp', 'SpMAD_Dzp', 'LogEE_Dzp', 'SM1_Dzp', 'VE1_Dzp', 'VE2_Dzp', 'VE3_Dzp', 'VR1_Dzp', 'VR2_Dzp', 'VR3_Dzp', 'SpAbs_Dzi', 'SpMax_Dzi', 'SpDiam_Dzi', 'SpAD_Dzi', 'SpMAD_Dzi', 'LogEE_Dzi', 'SM1_Dzi', 'VE1_Dzi', 'VE2_Dzi', 'VE3_Dzi', 'VR1_Dzi']]

y = df['SeverityClass']

print(X)
print(y)

# df = pd.read_csv('/Users/pantelispanka/Desktop/every-day/datasets/gdp-countries.csv')

lm = LinearRegression()
#
# y = df['GDP']
# X = df[['LFG', 'EQP', 'NEQ', 'GAP']]
#
model = lm.fit(X=X, y=y)
#
# print(model.predict(X))



jaqpot = Jaqpot()

jaqpot.set_api_key("eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ3Ujh3X1lGOWpKWFRWQ2x2VHF1RkswZkctQXROQUJsb3FBd0N4MmlTTWQ4In0.eyJleHAiOjE2MTcyMjI4NTgsImlhdCI6MTYxNzIxNTc2MiwiYXV0aF90aW1lIjoxNjE3MjE1NjU4LCJqdGkiOiIyYmIzZDE1Ni00YmU4LTQ4MmYtOTNkYi1hOTY0N2ZiMTE2NDMiLCJpc3MiOiJodHRwczovL2xvZ2luLmphcXBvdC5vcmcvYXV0aC9yZWFsbXMvamFxcG90IiwiYXVkIjoiYWNjb3VudCIsInN1YiI6IjI0MjVkNzYwLTAxOGQtNDA4YS1hZTBiLWNkZTRjNTYzNTRiOSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImphcXBvdC11aS1jb2RlIiwibm9uY2UiOiIwMzMwNzk2ZGQ1ZjAzYjI0MDQxZmI4NWNmYmYzZGJlNDc1Vmx6ODBiaiIsInNlc3Npb25fc3RhdGUiOiJkNzliNTc3Yi0wOWM2LTQ4NzUtYmRlYS1mZDk2NThhYWM3MzEiLCJhY3IiOiIwIiwiYWxsb3dlZC1vcmlnaW5zIjpbIicqJyIsIioiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBqYXFwb3QtYWNjb3VudHMgZW1haWwgcHJvZmlsZSB3cml0ZSByZWFkIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJuYW1lIjoiUGFudGVsaXMgS2FyYXR6YXMiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJwYW50ZWxpc3BhbmthIiwiZ2l2ZW5fbmFtZSI6IlBhbnRlbGlzIiwiZmFtaWx5X25hbWUiOiJLYXJhdHphcyIsImVtYWlsIjoicGFudGVsaXNwYW5rYUBnbWFpbC5jb20ifQ.WROBoJEFa1XuHs8c2piagTsnnb5z_i6Jn_ZWUyFzgQNsNQ9eTgy16RutLtaY7ZEe_Ucz6wv_Nk9NxCITOxKIJntEdhTx67ihRNSvme9pUbDtQqbnCVAL3rUB4DyHq-EgkNtzpxFpnWZRhF44x6dvSb7wkEdCHC99kPP9EfFhKXbcph1UrPr1f_hn2tYSSyU8kwn7e4wn1f-uR0ktVJiu7L_6FC-T6hXgc57kIArv2oEddAYO3A-Egpf5-65MSevGU8B0sAeULqXLmijVdQvrWrhGtBNHMZGo-wdvwrU2NhgS2RLuvA0NMexQNA4LwWiAsreavzeP6yWks5mMNik-Dg")

jaqpot.deploy_sklearn(model, X, y, title="Model with mordred 3", description="Test chempot", doa=X, model_meta=True)

jaqpot.deploy_sklearn_unsupervised()


# jaqpot.request_key_safe()