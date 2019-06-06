class Doa(object):

    def __init__(self, modelId=None, doaMatrix=None, aValue=None):
        self.modelId = None
        self.doaMatrix = None
        self.aValue = None

        if modelId is not None:
            self.modelId = modelId
        if doaMatrix is not None:
            self.doiMatrix = doaMatrix
        if aValue is not None:
            self.aValue = aValue
