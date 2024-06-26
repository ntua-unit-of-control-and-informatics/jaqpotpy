class PretrainedRequirements:

    def __init__(self, rawModel=None, independentFeatures=None,
                 predictedFeatures=None, additionalInfo=None,
                 title=None, implementedWith=None,
                 algorithm=None, implementedIn=None,
                 description=None, dependentFeatures=None
                 , jaqpotpyVersion=None, jaqpotpyDockerVersion=None, type=None, libraries=None, versions=None):
        self.rawModel = rawModel
        self.independentFeatures = independentFeatures
        self.predictedFeatures = predictedFeatures
        self.dependentFeatures = dependentFeatures
        self.additionalInfo = additionalInfo
        self.title = title
        self.implementedWith = implementedWith
        self.implementedIn = implementedIn
        self.algorithm = algorithm
        self.description = description
        self.jaqpotpyVersion = jaqpotpyVersion
        self.jaqpotpyDockerVersion = jaqpotpyDockerVersion
        self.type = type
        self.libraries = libraries
        self.versions = versions

        # if rawModel is not None:
        #     self.rawModel = rawModel
        # if independentFeatures is not None:
        #     self.independentFeatures = independentFeatures
        # if predictedFeatures is not None:
        #     self.predictedFeatures = predictedFeatures
        # if title is not None:
        #     self.title = title
        # if implementedWith is not None:
        #     self.implementedWith = implementedWith
        # if algorithm is not None:
        #     self.algorithm = algorithm
        # if implementedIn is not None:
        #     self.implementedIn = implementedIn
        # if description is not None:
        #     self.description = description
        # if dependentFeatures is not None:
        #     self.dependentFeatures = dependentFeatures

class PretrainedRequirements_v2:

    def __init__(self, id=None, meta=None, name=None, description=None, type=None,
                 jaqpotpyVersion=None, libraries=None, dependentFeatures=None,
                 independentFeatures=None, organizations=None, visibility=None,
                 reliability=None, pretrained=None, actualModel=None, creator=None,
                 canEdit=None, createdAt=None, updatedAt=None):
        self.id = id
        self.meta = meta
        self.name = name
        self.description = description
        self.type = type
        self.jaqpotpyVersion = jaqpotpyVersion
        self.libraries = libraries
        self.dependentFeatures = dependentFeatures
        self.independentFeatures = independentFeatures
        self.organizations = organizations
        self.visibility = visibility
        self.reliability = reliability
        self.pretrained = pretrained
        self.actualModel = actualModel
        self.creator = creator
        self.canEdit = canEdit
        self.createdAt = createdAt
        self.updatedAt = updatedAt