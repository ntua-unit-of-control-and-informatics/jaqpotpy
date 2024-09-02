class PretrainedRequirements:
    def __init__(
        self,
        rawModel=None,
        independentFeatures=None,
        predictedFeatures=None,
        additionalInfo=None,
        title=None,
        implementedWith=None,
        algorithm=None,
        implementedIn=None,
        description=None,
        dependentFeatures=None,
        jaqpotpyVersion=None,
        jaqpotpyDockerVersion=None,
        type=None,
        libraries=None,
        versions=None,
    ):
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
