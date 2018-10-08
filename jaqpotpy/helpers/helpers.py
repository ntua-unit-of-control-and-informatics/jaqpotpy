from jaqpotpy.entities.feature import Feature
from jaqpotpy.helpers.builders import FeatureBuilder,\
    FeatureDirector, DataEntryBuilder, DataEntryDirector,\
    PretrainedNeedsDirector, PretrainedNeedsBuilder


def create_feature(feat_title, creator):
    fb = FeatureBuilder()
    fb.add_comments("Feature created from python client for dataset creation")
    fb.add_creator(creator)
    fb.add_title(feat_title)
    dire = FeatureDirector()
    mf = dire.construct(fb)
    return mf


def clear_entity(o):
    delete = []
    for key in o:
        if getattr(o, key) is None or "":
            delete.append(key)
    for keytod in delete:
        delattr(o, keytod)
    return o


def create_data_entry(df, feat_map, owner_uuid):
    data_entry = []
    for dataid in df.index:
        data_entry_all = {}
        values_from_dataframe = df.loc[dataid]
        de_director = DataEntryDirector()
        de_builder = DataEntryBuilder()
        de_builder.set_name(dataid)
        de_builder.set_owneruuid(owner_uuid)
        values = {}
        # print(feat_map)
        for key in feat_map:
            # print(key)
            # print(feat_map[key])
            # print(values_from_dataframe[key])
            values[feat_map[key]] = values_from_dataframe[key]
        de_builder.set_values(values)
        de = de_director.construct(de_builder)
        data_entry.append(de.__dict__)
    return data_entry


def create_pretrain_req(model, X, y, title, description, algorithm, implementedWith, implementedIn, additionalInfo):
    pnb = PretrainedNeedsBuilder()
    independentFeatures = []
    dependendFeatures = []
    for fe in list(X):
        independentFeatures.append(fe)
    if type(y).__name__ is 'Dataframe':
        for fe in list(y):
            dependendFeatures.append(fe)
    elif type(y).__name__ is 'Series':
        dependendFeatures.append(y.name)
    pnb.setRawModel(model)
    pnb.setAdditionalInfo(additionalInfo)
    pnb.setAlgorithm(algorithm)
    pnb.setDependendFeatures(dependendFeatures)
    pnb.setIndependentFeatures(independentFeatures)
    pnb.setDescription(description)
    pnb.setTitle(title)
    pnb.setImplementedIn(implementedIn)
    pnb.setImplementedWith(implementedWith)
    director = PretrainedNeedsDirector()
    return director.construct(pnb)
