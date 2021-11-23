from jaqpotpy import Jaqpot
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# df = pd.read_csv('/Users/pantelispanka/Data/SMILES/ltkb_all_mordred.csv')
#
# print(list(df))

# X = df[['SpMax_Dzare', 'SpDiam_Dzare', 'SpAD_Dzare', 'SpMAD_Dzare', 'LogEE_Dzare', 'SM1_Dzare', 'VE1_Dzare', 'VE2_Dzare', 'VE3_Dzare', 'VR1_Dzare', 'VR2_Dzare', 'VR3_Dzare', 'SpAbs_Dzp', 'SpMax_Dzp', 'SpDiam_Dzp', 'SpAD_Dzp', 'SpMAD_Dzp', 'LogEE_Dzp', 'SM1_Dzp', 'VE1_Dzp', 'VE2_Dzp', 'VE3_Dzp', 'VR1_Dzp', 'VR2_Dzp', 'VR3_Dzp', 'SpAbs_Dzi', 'SpMax_Dzi', 'SpDiam_Dzi', 'SpAD_Dzi', 'SpMAD_Dzi', 'LogEE_Dzi', 'SM1_Dzi', 'VE1_Dzi', 'VE2_Dzi', 'VE3_Dzi', 'VR1_Dzi']]
# X = df[['MINssAsH', 'MINsssAs', 'MINsssdAs', 'MINsssssAs', 'MINsSeH', 'MINdSe', 'MINssSe', 'MINaaSe', 'MINdssSe', 'MINddssSe', 'MINsBr', 'MINsSnH3', 'MINssSnH2', 'MINsssSnH', 'MINssssSn', 'MINsI', 'MINsPbH3', 'MINssPbH2', 'MINsssPbH', 'MINssssPb', 'ECIndex', 'ETA_alpha', 'AETA_alpha', 'ETA_shape_p', 'ETA_shape_y', 'ETA_shape_x', 'ETA_beta', 'AETA_beta']]
# X = df[['PetitjeanIndex', 'Vabc', 'VAdjMat', 'MWC01', 'MWC02', 'MWC03', 'MWC04', 'MWC05', 'MWC06', 'MWC07', 'MWC08', 'MWC09', 'MWC10', 'TMWC10', 'SRW02', 'SRW03', 'SRW04', 'SRW05', 'SRW06', 'SRW07', 'SRW08', 'SRW09', 'SRW10', 'TSRW10', 'MW', 'AMW', 'WPath', 'WPol', 'Zagreb1', 'Zagreb2', 'mZagreb1']]
# y = df['vDILIConcern']

# y = df['SeverityClass']

# print(X)
# print(y)

# df = pd.read_csv('/Users/pantelispanka/Desktop/every-day/datasets/gdp-countries.csv')

# lm = LinearRegression()
# model = lm.fit(X=X, y=y)
#
# y = df['GDP']
# X = df[['LFG', 'EQP', 'NEQ', 'GAP']]
#

#
# print(model.predict(X))

# X = df[['MINssAsH', 'MINsssAs', 'MINsssdAs', 'MINsssssAs', 'MINsSeH', 'MINdSe', 'MINssSe', 'MINaaSe', 'MINdssSe', 'MINddssSe', 'MINsBr', 'MINsSnH3', 'MINssSnH2', 'MINsssSnH', 'MINssssSn', 'MINsI', 'MINsPbH3', 'MINssPbH2', 'MINsssPbH', 'MINssssPb', 'ECIndex', 'ETA_alpha', 'AETA_alpha', 'ETA_shape_p', 'ETA_shape_y', 'ETA_shape_x', 'ETA_beta', 'AETA_beta']]

# y = df['SeverityClass']

# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=8, random_state=0))

# clf.fit(X, y)

# jaqpot = Jaqpot()

# jaqpot = Jaqpot("https://squonkpotapi.jaqpot.org/jaqpot/services/")

# jaqpot.login('ppanka','kapan1')

# jaqpot.set_api_key("eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI3RC1USHRaTVdNRElhV3gxX0NXamhCVHpWdEpaejRBTnd2dGh3QWx2OFRrIn0.eyJleHAiOjE2MzM0NTQ5MzAsImlhdCI6MTYzMzQzMzMzMCwiYXV0aF90aW1lIjoxNjMzNDMzMzI5LCJqdGkiOiJmMjdhZWM3Yi01NjA1LTQ4MGQtOTQzNy05NzY4YzE5OTJkYzUiLCJpc3MiOiJodHRwczovL3NxdW9uay5pbmZvcm1hdGljc21hdHRlcnMub3JnL2F1dGgvcmVhbG1zL3NxdW9uayIsImF1ZCI6WyJzcXVvbmstam9iZXhlY3V0b3IiLCJzcXVvbmstcG9ydGFsIiwiYWNjb3VudCJdLCJzdWIiOiI4NWE4NzIyMS01YWMxLTQyYWUtOWRlMC05NDliZjdkMzNhM2UiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJqYXFwb3QtdWkiLCJub25jZSI6IjY4NDRmODM4N2NiYmFhOTA2NWJhYmJkMTI1ZDgxYmYyZDBrdXlrWjN5Iiwic2Vzc2lvbl9zdGF0ZSI6ImVlMWI3MGNjLWZlZjUtNDlkYy1iMGU4LTk3MDNjYjgyYjFhMCIsImFjciI6IjEiLCJhbGxvd2VkLW9yaWdpbnMiOlsiaHR0cHM6Ly9zcXVvbnBvdC5qYXFwb3Qub3JnIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJzdGFuZGFyZC11c2VyIiwiZGF0YS1tYW5hZ2VyLXVzZXIiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsic3F1b25rLWpvYmV4ZWN1dG9yIjp7InJvbGVzIjpbInN0YW5kYXJkLXVzZXIiXX0sInNxdW9uay1wb3J0YWwiOnsicm9sZXMiOlsic3RhbmRhcmQtdXNlciJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBlbWFpbCIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwibmFtZSI6IlBhbnRlbGlzIEthcmF0emFzIiwicHJlZmVycmVkX3VzZXJuYW1lIjoicHBhbmthIiwiZ2l2ZW5fbmFtZSI6IlBhbnRlbGlzIiwiZmFtaWx5X25hbWUiOiJLYXJhdHphcyIsImVtYWlsIjoicGFudGVsaXNwYW5rYUBnbWFpbC5jb20ifQ.Yf-7D_ZXFf6afTMEfRxRZuD7e2KxKjcUqvXPWoEwk3TsxqL31K6b9m70ilQk-AVTXw3cym0yFKq5Zt3gSvyQ3gQn10Sji7K9l3CochjTIwkDFjuXUkif0PFFINKN9NYlAFgjNhB8xS-1AKput6UiSFHH5que09Yimsbu6kwIzdTc45yJzZVPVL-A3c-Ec59URnrbk6ozIBj_rcIoGBwsgAc8r7IzQE4kvI4CyGI56642g4jiD59tYc-GeEoX5Jv5K7JI_x8mqFe3GQhKf2ymj_J8p3uPRcSO3Sx1lww_oaumQpFyzvmL6qPdf0BvB1P05nfwSFteFFSHWhgT0yy7OQ")


# jaqpot.deploy_sklearn(clf, X, y, title="Predicting Dili severity with Random forests", description="Test chempot 3")
# jaqpot = Jaqpot()
# jaqpot.request_key_safe()
# jaqpot.deploy_sklearn(clf, X, y, title="Predicting Dili severity with Random forests", description="Test chempot 3")
# jaqpot = Jaqpot("http://localhost:8080/jaqpot/services/")
# jaqpot.request_key_safe()
# jaqpot.set_api_key("eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ3Ujh3X1lGOWpKWFRWQ2x2VHF1RkswZkctQXROQUJsb3FBd0N4MmlTTWQ4In0.eyJleHAiOjE2MTcyMjI4NTgsImlhdCI6MTYxNzIxNTc2MiwiYXV0aF90aW1lIjoxNjE3MjE1NjU4LCJqdGkiOiIyYmIzZDE1Ni00YmU4LTQ4MmYtOTNkYi1hOTY0N2ZiMTE2NDMiLCJpc3MiOiJodHRwczovL2xvZ2luLmphcXBvdC5vcmcvYXV0aC9yZWFsbXMvamFxcG90IiwiYXVkIjoiYWNjb3VudCIsInN1YiI6IjI0MjVkNzYwLTAxOGQtNDA4YS1hZTBiLWNkZTRjNTYzNTRiOSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImphcXBvdC11aS1jb2RlIiwibm9uY2UiOiIwMzMwNzk2ZGQ1ZjAzYjI0MDQxZmI4NWNmYmYzZGJlNDc1Vmx6ODBiaiIsInNlc3Npb25fc3RhdGUiOiJkNzliNTc3Yi0wOWM2LTQ4NzUtYmRlYS1mZDk2NThhYWM3MzEiLCJhY3IiOiIwIiwiYWxsb3dlZC1vcmlnaW5zIjpbIicqJyIsIioiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBqYXFwb3QtYWNjb3VudHMgZW1haWwgcHJvZmlsZSB3cml0ZSByZWFkIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJuYW1lIjoiUGFudGVsaXMgS2FyYXR6YXMiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJwYW50ZWxpc3BhbmthIiwiZ2l2ZW5fbmFtZSI6IlBhbnRlbGlzIiwiZmFtaWx5X25hbWUiOiJLYXJhdHphcyIsImVtYWlsIjoicGFudGVsaXNwYW5rYUBnbWFpbC5jb20ifQ.WROBoJEFa1XuHs8c2piagTsnnb5z_i6Jn_ZWUyFzgQNsNQ9eTgy16RutLtaY7ZEe_Ucz6wv_Nk9NxCITOxKIJntEdhTx67ihRNSvme9pUbDtQqbnCVAL3rUB4DyHq-EgkNtzpxFpnWZRhF44x6dvSb7wkEdCHC99kPP9EfFhKXbcph1UrPr1f_hn2tYSSyU8kwn7e4wn1f-uR0ktVJiu7L_6FC-T6hXgc57kIArv2oEddAYO3A-Egpf5-65MSevGU8B0sAeULqXLmijVdQvrWrhGtBNHMZGo-wdvwrU2NhgS2RLuvA0NMexQNA4LwWiAsreavzeP6yWks5mMNik-Dg")
# jaqpot.deploy_sklearn(model, X, y, title="Model with mordred 0", description="Test chempot", doa=X, model_meta=True)
# jaqpot.deploy_sklearn_unsupervised()
# jaqpot.request_key_safe()


# jaqpot = Jaqpot("https://squonkpotapi.jaqpot.org/jaqpot/services/")
# jaqpot.set_api_key("eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI3RC1USHRaTVdNRElhV3gxX0NXamhCVHpWdEpaejRBTnd2dGh3QWx2OFRrIn0.eyJleHAiOjE2MzM0NTQ5MzAsImlhdCI6MTYzMzQzMzMzMCwiYXV0aF90aW1lIjoxNjMzNDMzMzI5LCJqdGkiOiJmMjdhZWM3Yi01NjA1LTQ4MGQtOTQzNy05NzY4YzE5OTJkYzUiLCJpc3MiOiJodHRwczovL3NxdW9uay5pbmZvcm1hdGljc21hdHRlcnMub3JnL2F1dGgvcmVhbG1zL3NxdW9uayIsImF1ZCI6WyJzcXVvbmstam9iZXhlY3V0b3IiLCJzcXVvbmstcG9ydGFsIiwiYWNjb3VudCJdLCJzdWIiOiI4NWE4NzIyMS01YWMxLTQyYWUtOWRlMC05NDliZjdkMzNhM2UiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJqYXFwb3QtdWkiLCJub25jZSI6IjY4NDRmODM4N2NiYmFhOTA2NWJhYmJkMTI1ZDgxYmYyZDBrdXlrWjN5Iiwic2Vzc2lvbl9zdGF0ZSI6ImVlMWI3MGNjLWZlZjUtNDlkYy1iMGU4LTk3MDNjYjgyYjFhMCIsImFjciI6IjEiLCJhbGxvd2VkLW9yaWdpbnMiOlsiaHR0cHM6Ly9zcXVvbnBvdC5qYXFwb3Qub3JnIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJzdGFuZGFyZC11c2VyIiwiZGF0YS1tYW5hZ2VyLXVzZXIiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsic3F1b25rLWpvYmV4ZWN1dG9yIjp7InJvbGVzIjpbInN0YW5kYXJkLXVzZXIiXX0sInNxdW9uay1wb3J0YWwiOnsicm9sZXMiOlsic3RhbmRhcmQtdXNlciJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBlbWFpbCIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwibmFtZSI6IlBhbnRlbGlzIEthcmF0emFzIiwicHJlZmVycmVkX3VzZXJuYW1lIjoicHBhbmthIiwiZ2l2ZW5fbmFtZSI6IlBhbnRlbGlzIiwiZmFtaWx5X25hbWUiOiJLYXJhdHphcyIsImVtYWlsIjoicGFudGVsaXNwYW5rYUBnbWFpbC5jb20ifQ.Yf-7D_ZXFf6afTMEfRxRZuD7e2KxKjcUqvXPWoEwk3TsxqL31K6b9m70ilQk-AVTXw3cym0yFKq5Zt3gSvyQ3gQn10Sji7K9l3CochjTIwkDFjuXUkif0PFFINKN9NYlAFgjNhB8xS-1AKput6UiSFHH5que09Yimsbu6kwIzdTc45yJzZVPVL-A3c-Ec59URnrbk6ozIBj_rcIoGBwsgAc8r7IzQE4kvI4CyGI56642g4jiD59tYc-GeEoX5Jv5K7JI_x8mqFe3GQhKf2ymj_J8p3uPRcSO3Sx1lww_oaumQpFyzvmL6qPdf0BvB1P05nfwSFteFFSHWhgT0yy7OQ")


# jaqpot = Jaqpot("https://modelsbase.cloud.nanosolveit.eu/modelsbase/services/")
# jaqpot.request_key_safe()


# model = jaqpot.get_model_by_id("owfIWJ0eYuYKpws0CmFc")
# print(model)



# pip install jaqpotpy


# jaqpot = Jaqpot()
# jaqpot.set_api_key("eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ3Ujh3X1lGOWpKWFRWQ2x2VHF1RkswZkctQXROQUJsb3FBd0N4MmlTTWQ4In0.eyJleHAiOjE2MzM0NTM4ODEsImlhdCI6MTYzMzQ0NjY4MSwiYXV0aF90aW1lIjoxNjMzNDMzNjY5LCJqdGkiOiJmZTIxZTY4ZS0yZmNiLTQ1ODMtOWZmMi0yZGMxZTNhZDk5NGUiLCJpc3MiOiJodHRwczovL2xvZ2luLmphcXBvdC5vcmcvYXV0aC9yZWFsbXMvamFxcG90IiwiYXVkIjoiYWNjb3VudCIsInN1YiI6IjI0MjVkNzYwLTAxOGQtNDA4YS1hZTBiLWNkZTRjNTYzNTRiOSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImphcXBvdC11aS1jb2RlIiwibm9uY2UiOiI5ZDllYTFlMGVmMDkxZWFiMjgxYjBjZGEyODQ1Y2ZmMDkwV0ZGYmxDaSIsInNlc3Npb25fc3RhdGUiOiI0ZWRhNWZmYi03MDI5LTQwZjEtOGZkMi0wMTdkY2Y0OWMyYTkiLCJhY3IiOiIwIiwiYWxsb3dlZC1vcmlnaW5zIjpbIicqJyIsIioiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBqYXFwb3QtYWNjb3VudHMgZW1haWwgcHJvZmlsZSB3cml0ZSByZWFkIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJuYW1lIjoiUGFudGVsaXMgS2FyYXR6YXMiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJwYW50ZWxpc3BhbmthIiwiZ2l2ZW5fbmFtZSI6IlBhbnRlbGlzIiwiZmFtaWx5X25hbWUiOiJLYXJhdHphcyIsImVtYWlsIjoicGFudGVsaXNwYW5rYUBnbWFpbC5jb20ifQ.UJKTvLxqSPzV5r3KnI3vc2ZJkuJqsrrygNl343IMc6LXJsUuXUCBmbx2t3paAM5GgN8qnpgsbu7yPHcYQhuyyQhjSvOUWPAY8nnHQZGqRBeX_sRacWY6BBESBBkrNI7MXyOW4Q35D8a3S5nGLXVUjNPiMXeu54boSVpGNVBPJIj-mAd02oHO8EtMsQW9XOEJXwV-BEN2cCvnbURwLmh4pd7Mauet03_icqyFbu7em-_bXAYMpTvzP9buQK4BRQNrzL4fvbud3U2TA9JJs5ahmK1q_c74tKFxQPWAGhZk6M_MJjTHKWaTW6IYimTzrYZyDOTlMYqWxncQ67xK-juc5w")

# df, predicts = jaqpot.chempot_predict("owfIWJ0eYuYKpws0CmFc", "C3CCC(C(C1CCCC1)C2CCCC2)C3")


# df is the returned pandas dataframe with all the returned values and the predicted.
# predicts object is the column name that hold the prediction
# df, predicts = jaqpot.chempot_predict("MHBj2AL2xmhDTBIKm5RP", "C3CCC(C(C1CCCC1)C2CCCC2)C3")




# df, predicts = jaqpot.chempot_predict("rF4f4poSPs1W8Bdazic9", "C3CCC(C(C1CCCC1)C2CCCC2)C3")

#
# print(df)
# print(predicts)

from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("/Users/pantelispanka/Downloads/PCF_filtered_by_gsva.csv")
print(list(df))

bio = df[['P01024', 'P0C0L4', 'P02649', 'P10909', 'Q14624', 'P01009', 'P04114', 'P00734', 'P0C0L5', 'P01008', 'P04196', 'P01023', 'P01042', 'P02656', 'P00739', 'P05154', 'P02743', 'P06396', 'P19823', 'P12259', 'P10720', 'P05546', 'P49908', 'P35542', 'P68871', 'Q03591', 'O43866', 'P02749', 'P03951', 'P02654', 'P03952', 'P02760', 'P00738', 'P01011', 'P18428', 'P02655', 'Q13103', 'P00736', 'P00748', 'P00742', 'P02774', 'Q14520', 'P00751', 'P00740', 'P03950', 'P02790', 'P09871', 'P27169', 'P02788', 'P20851', 'P18065', 'P00450', 'P08567', 'P01019', 'P02671', 'P15169', 'Q13790', 'P08709', 'P00451', 'Q06033', 'P14618', 'P23528', 'Q99467']]

sim = cosine_similarity(bio)
print(sim[0])
i = 0
for j in sim[0]:
    if j > 0.8:
        i += 1
print(i)
# print(bio)
# print(df['pdi_synth'])
