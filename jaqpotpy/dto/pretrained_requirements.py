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
