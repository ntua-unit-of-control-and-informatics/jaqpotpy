import abc
from jaqpotpy.entities.feature import Feature
from jaqpotpy.entities.meta import MetaInfo
from jaqpotpy.entities.entryid import EntryId
from jaqpotpy.entities.dataentry import DataEntry
from jaqpotpy.dto.pretrained_requirements import PretrainedRequirements
from jaqpotpy.entities.doa import Doa
from base64 import b64encode
import pickle


class FeatureDirector:

    def __init__(self):
        self._builder = None

    def construct(self, builder):
        meta_info = MetaInfo()
        self._builder = builder
        meta_info.comments = self._builder.get_comments()
        meta_info.descriptions = self._builder.get_descriptions()
        meta_info.titles = self._builder.get_title()
        meta_info.creators = self._builder.get_creator()
        feature = Feature()
        feature.meta = meta_info.__dict__
        feature.units = self._builder.get_units()
        return feature


class FBuilder(metaclass=abc.ABCMeta):

    def __init__(self):
        self.metainfo = MetaInfo()

    @abc.abstractmethod
    def add_comments(self, comment):
        pass

    @abc.abstractmethod
    def add_descriptions(self, description):
        pass

    @abc.abstractmethod
    def add_title(self, title):
        pass

    @abc.abstractmethod
    def add_creator(self, creator):
        pass

    @abc.abstractmethod
    def set_units(self, units):
        pass

    @abc.abstractmethod
    def get_comments(self):
        pass

    @abc.abstractmethod
    def get_descriptions(self):
        pass

    @abc.abstractmethod
    def get_title(self):
        pass

    @abc.abstractmethod
    def get_creator(self):
        pass

    @abc.abstractmethod
    def get_units(self):
        pass


class FeatureBuilder(FBuilder):
    comments = []
    descriptions = []
    titles = []
    creators = []
    units = None

    def add_comments(self, comment):
        self.comments.clear()
        self.comments.append(comment)

    def add_descriptions(self, description):
        self.descriptions.clear()
        self.descriptions.append(description)

    def add_title(self, title):
        self.titles.clear()
        self.titles.append(title)

    def add_creator(self, creator):
        self.creators.clear()
        self.creators.append(creator)

    def set_units(self, units):
        self.units = units

    def get_comments(self):
        return self.comments

    def get_descriptions(self):
        return self.descriptions

    def get_title(self):
        return self.titles

    def get_creator(self):
        return self.creators

    def get_units(self):
        return self.units


class DataEntryDirector:

    def __init__(self):
        self._builder = None

    def construct(self, builder):
        entryId = EntryId()
        self._builder = builder
        entryId.name = self._builder.get_name()
        entryId.ownerUUID = self._builder.get_owneruuid()
        entryId.type = self._builder.get_type()
        entryId.URI = self._builder.get_uri()
        data_entry = DataEntry()
        data_entry.entryId = entryId.__dict__
        data_entry.values = self._builder.get_values()
        return data_entry


class DBuilder(metaclass=abc.ABCMeta):

    def __init__(self):
        self.dataentry = EntryId()

    @abc.abstractmethod
    def set_name(self, name):
        pass

    @abc.abstractmethod
    def set_owneruuid(self, owner_uuid):
        pass

    @abc.abstractmethod
    def set_type(self, type):
        pass

    @abc.abstractmethod
    def set_uri(self, uri):
        pass

    @abc.abstractmethod
    def set_values(self, value):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    @abc.abstractmethod
    def get_owneruuid(self):
        pass

    @abc.abstractmethod
    def get_type(self):
        pass

    @abc.abstractmethod
    def get_uri(self):
        pass

    @abc.abstractmethod
    def get_values(self):
        pass


class DataEntryBuilder(DBuilder):

    name = None
    ownerr_uuid = None
    type = None
    uri = None
    value = None

    def set_name(self, name):
        self.name = name

    def set_owneruuid(self, owner_uuid):
        self.ownerr_uuid = owner_uuid

    def set_type(self, type):
        self.type = type

    def set_uri(self, uri):
        self.uri = uri

    def set_values(self, value):
        self.value = value

    def get_name(self):
        return self.name

    def get_owneruuid(self):
        return self.ownerr_uuid

    def get_type(self):
        return self.type

    def get_uri(self):
        return self.uri

    def get_values(self):
        return self.value


class PretrainedNeedsDirector:

    def __init__(self):
        self._builder = None

    def construct(self, builder):
        pretrained_needs = PretrainedRequirements()
        self._builder = builder
        pretrained_needs.rawModel = self._builder.getRawModel()
        pretrained_needs.independentFeatures = self._builder.getIndependentFeatures()
        pretrained_needs.dependentFeatures = self._builder.getDependendFeatures()
        pretrained_needs.predictedFeatures = self._builder.getDependendFeatures()
        pretrained_needs.implementedWith = self._builder.getImplementedWith()
        pretrained_needs.additionalInfo = self._builder.getAdditionalInfo()
        pretrained_needs.algorithm = self._builder.getAlgorithm()
        pretrained_needs.runtime = self._builder.getRuntime()
        pretrained_needs.description = self._builder.getDescription()
        pretrained_needs.title = self._builder.getTitle()
        pretrained_needs.jaqpotpyVersion = self._builder.getJaqpotPyVersion()
        pretrained_needs.jaqpotpyDockerVersion = self._builder.getJaqpotpyDockerVersion()
        pretrained_needs.libraries = self._builder.getLibraries()
        pretrained_needs.versions = self._builder.getVersions()
        pretrained_needs.type = self._builder.getType()
        pretrained_needs.runtime = self._builder.getRuntime()
        return pretrained_needs


class PretrainedNeedsBuilder:
    ENCODING = 'utf-8'
    independendFeatures = None
    dependendFeatures = None
    implementedWith = []
    additionalInfo = None
    algorithm = []
    runtime = []
    description = []
    title = []
    rawModel = []
    libraries = []
    versions = []
    jaqpotpy_version: str = None
    jaqpotpy_docker_version: str = None
    type: str = None

    def setLibraries(self, libraries):
        self.libraries.clear()
        self.libraries = libraries

    def setVersions(self, versions):
        self.versions.clear()
        self.versions = versions

    def setJaqpotPyVersion(self, jaqpotpy_version):
        # self.jaqpotpy_version.clear()
        self.jaqpotpy_version = jaqpotpy_version

    def setJaqpotPyDockerVersion(self, jaqpotpy_docker_version):
        # self.jaqpotpy_version.clear()
        self.jaqpotpy_docker_version = jaqpotpy_docker_version

    def setType(self, type):
        # self.type.clear()
        self.type = type

    def getType(self):
        # self.type.clear()
        return self.type

    def setRawModel(self, model):
        self.rawModel.clear()
        p_mod = pickle.dumps(model)
        raw = b64encode(p_mod)
        raw_model = raw.decode(self.ENCODING)
        # self.rawModel = raw_model
        self.rawModel.append(raw_model)

    def setIndependentFeatures(self, ind_f):
        self.independendFeatures = ind_f

    def setDependendFeatures(self, dep_f):
        self.dependendFeatures = dep_f

    def setImplementedWith(self, impWith):
        self.implementedWith.clear()
        self.implementedWith.append(impWith)

    def setAdditionalInfo(self, add_inf):
        self.additionalInfo = add_inf

    def setAlgorithm(self, algo):
        self.algorithm.clear()
        self.algorithm.append(algo)

    """Sets theImplementedIn of this Pretrained model.
        It correspondes to the runtime that is needed for the algorithm to run
       #
       #
       #     :param implementedIn: The runtime of this Model.
       #     """

    def setRuntime(self, runtime):
        self.runtime.clear()
        self.runtime.append(runtime)

    def setDescription(self, descr):
        self.description.clear()
        self.description.append(descr)

    def setTitle(self, title):
        self.title.clear()
        self.title.append(title)

    def getRawModel(self):
        return self.rawModel

    def getIndependentFeatures(self):
        return self.independendFeatures

    def getDependendFeatures(self):
        return self.dependendFeatures

    def getImplementedWith(self):
        return self.implementedWith

    def getAdditionalInfo(self):
        return self.additionalInfo

    def getAlgorithm(self):
        return self.algorithm

    def getRuntime(self):
        return self.runtime

    def getDescription(self):
        return self.description

    def getTitle(self):
        return self.title

    def getJaqpotPyVersion(self):
        return self.jaqpotpy_version

    def getJaqpotpyDockerVersion(self):
        return self.jaqpotpy_docker_version

    def getLibraries(self):
        return self.libraries

    def getVersions(self):
        return self.versions


class DOABuilder(metaclass=abc.ABCMeta):

    def __init__(self):
        self.doa = Doa()

    @abc.abstractmethod
    def set_modelid(self, modelid):
        pass

    @abc.abstractmethod
    def set_doa_matrix(self, owner_uuid):
        pass

    @abc.abstractmethod
    def set_aValue(self, a):
        pass

    @abc.abstractmethod
    def get_modelid(self):
        pass

    @abc.abstractmethod
    def get_doa_matrix(self):
        pass

    @abc.abstractmethod
    def get_aValue(self):
        pass


class DoaBuilder(DOABuilder):
    modelId = None
    doaMatrix = None
    aValue = None

    def set_modelid(self, modelid):
        self.modelId = modelid

    def set_doa_matrix(self, doa_matrix):
        self.doaMatrix = doa_matrix

    def set_aValue(self, aValue):
        self.aValue = aValue

    def get_modelid(self):
        return self.modelId

    def get_doa_matrix(self):
        return self.doaMatrix

    def get_aValue(self):
        return self.aValue


class DoaDirector:
    def __init__(self):
        self._builder = None

    def construct(self, builder):
        doa = Doa()
        self._builder = builder
        doa.modelId = self._builder.get_modelid()
        doa.doaMatrix = self._builder.get_doa_matrix()
        doa.aValue = self._builder.get_aValue()
        return doa


# mb = MetaFeatureBuilder()
# mb.add_comments("Comments")
# mb.add_creator("Creator")
# mb.add_descriptions("Descr")
# mb.add_title("Title")
#
# dire = MetaDirector()
# mf = dire.construct(mb)
#
# print(mf.descriptions, mf.titles, mf.comments)