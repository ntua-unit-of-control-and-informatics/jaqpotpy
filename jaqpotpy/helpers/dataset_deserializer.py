import json
import pandas as pd

def decode_predicted(dataset):
    feat_info = dataset['features']
    predicted = {}
    predicted_f = []
    for feat in feat_info:
        name = feat['name']
        key = feat['key']
        try:
            category = feat['category']
            if (category == 'PREDICTED') and (feat['name'] not in predicted_f):
                predicted_f.append(feat['name'])
        except KeyError:
            dataentries = {}
            for dataEntry in dataset['dataEntry']:
                dataentries[dataEntry['entryId']['name']] = dataEntry['values'][key]
            predicted[name] = dataentries
        dataentries = {}
        for dataEntry in dataset['dataEntry']:
            dataentries[dataEntry['entryId']['name']] = dataEntry['values'][key]
        predicted[name] = dataentries
    json_to_df = json.dumps(predicted)
    df = pd.read_json(json_to_df)
    return df, predicted_f

            # for val in dataEntry['values']:
#         for dataEntry in dataset['dataEntry']:
#             dataEntryToInsert = []
#             for key in shorted:
#                 dataEntryToInsert.append(dataEntry['values'][key])
#             dataEntryAll.append(dataEntryToInsert)
    #     for key in independentFeatures:
    #         if actual == independentFeatures[key]:
    #             for feature in pred_request.dataset['features']:
    #                 if feature['name'] == actual:
    #                     shorted.append(feature['key'])
    #             # shorted.append(key)
    # dataEntryAll = []
    # for dataEntry in pred_request.dataset['dataEntry']:
    #     dataEntryToInsert = []
    #     for key in shorted:
    #         dataEntryToInsert.append(dataEntry['values'][key])
    #     dataEntryAll.append(dataEntryToInsert)
    # return dataEntryAll