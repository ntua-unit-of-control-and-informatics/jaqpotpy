from .jaqpot_base import JaqpotEntity
from .dataentry import DataEntry
from .featureinfo import FeatureInfo
from .meta import MetaInfo


class Dataset(object):

    def __init__(self, meta=None, ontologicalClasses=None, visible=False, temporary=False, featured=False, datasetUri=None, byModel=None, dataEntry=None, features=None, totalRows=None, totalColumns=None, descriptors=None, id=None, existence=None):
        """Dataset - a model defined in Swagger"""  # noqa: E501

        self.meta = None
        self.ontologicalClasses = None
        self.visible = None
        self.temporary = None
        self.featured = None
        self.datasetUri = None
        self.byModel = None
        self.dataEntry = None
        self.features = None
        self.totalRows = None
        self.totalColumns = None
        self.descriptors = None
        self.id = None
        self.existence = None

        if meta is not None:
            self.meta = meta
        if ontologicalClasses is not None:
            self.ontologicalClasses = ontologicalClasses
        if visible is not None:
            self.visible = visible
        if temporary is not None:
            self.temporary = temporary
        if featured is not None:
            self.featured = featured
        if datasetUri is not None:
            self.dataset_uri = datasetUri
        if byModel is not None:
            self.by_model = byModel
        if dataEntry is not None:
            self.data_entry = dataEntry
        if features is not None:
            self.features = features
        if totalRows is not None:
            self.totalRows = totalRows
        if totalColumns is not None:
            self.totalColumns = totalColumns
        if descriptors is not None:
            self.descriptors = descriptors
        if id is not None:
            self.id = id
        if existence is not None:
            self.existence = existence

    # @property
    # def meta(self):
    #     """Gets the meta of this Dataset.  # noqa: E501
    #
    #
    #     :return: The meta of this Dataset.  # noqa: E501
    #     :rtype: MetaInfo
    #     """
    #     return self.meta
    #
    # @meta.setter
    # def meta(self, meta):
    #     """Sets the meta of this Dataset.
    #
    #
    #     :param meta: The meta of this Dataset.  # noqa: E501
    #     :type: MetaInfo
    #     """
    #
    #     self.meta = meta
    #
    # @property
    # def ontological_classes(self):
    #     """Gets the ontological_classes of this Dataset.  # noqa: E501
    #
    #
    #     :return: The ontological_classes of this Dataset.  # noqa: E501
    #     :rtype: list[str]
    #     """
    #     return self.ontological_classes
    #
    # @ontological_classes.setter
    # def ontological_classes(self, ontological_classes):
    #     """Sets the ontological_classes of this Dataset.
    #
    #
    #     :param ontological_classes: The ontological_classes of this Dataset.  # noqa: E501
    #     :type: list[str]
    #     """
    #
    #     self.ontological_classes = ontological_classes
    #
    # @property
    # def visible(self):
    #     """Gets the visible of this Dataset.  # noqa: E501
    #
    #
    #     :return: The visible of this Dataset.  # noqa: E501
    #     :rtype: bool
    #     """
    #     return self.visible
    #
    # @visible.setter
    # def visible(self, visible):
    #     """Sets the visible of this Dataset.
    #
    #
    #     :param visible: The visible of this Dataset.  # noqa: E501
    #     :type: bool
    #     """
    #
    #     self.visible = visible
    #
    # @property
    # def temporary(self):
    #     """Gets the temporary of this Dataset.  # noqa: E501
    #
    #
    #     :return: The temporary of this Dataset.  # noqa: E501
    #     :rtype: bool
    #     """
    #     return self.temporary
    #
    # @temporary.setter
    # def temporary(self, temporary):
    #     """Sets the temporary of this Dataset.
    #
    #
    #     :param temporary: The temporary of this Dataset.  # noqa: E501
    #     :type: bool
    #     """
    #
    #     self.temporary = temporary
    #
    # @property
    # def featured(self):
    #     """Gets the featured of this Dataset.  # noqa: E501
    #
    #
    #     :return: The featured of this Dataset.  # noqa: E501
    #     :rtype: bool
    #     """
    #     return self.featured
    #
    # @featured.setter
    # def featured(self, featured):
    #     """Sets the featured of this Dataset.
    #
    #
    #     :param featured: The featured of this Dataset.  # noqa: E501
    #     :type: bool
    #     """
    #
    #     self.featured = featured
    #
    # @property
    # def dataset_uri(self):
    #     """Gets the dataset_uri of this Dataset.  # noqa: E501
    #
    #
    #     :return: The dataset_uri of this Dataset.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self.dataset_uri
    #
    # @dataset_uri.setter
    # def dataset_uri(self, dataset_uri):
    #     """Sets the dataset_uri of this Dataset.
    #
    #
    #     :param dataset_uri: The dataset_uri of this Dataset.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self.dataset_uri = dataset_uri
    #
    # @property
    # def by_model(self):
    #     """Gets the by_model of this Dataset.  # noqa: E501
    #
    #
    #     :return: The by_model of this Dataset.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self.by_model
    #
    # @by_model.setter
    # def by_model(self, by_model):
    #     """Sets the by_model of this Dataset.
    #
    #
    #     :param by_model: The by_model of this Dataset.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self.by_model = by_model
    #
    # @property
    # def data_entry(self):
    #     """Gets the data_entry of this Dataset.  # noqa: E501
    #
    #
    #     :return: The data_entry of this Dataset.  # noqa: E501
    #     :rtype: list[DataEntry]
    #     """
    #     return self.data_entry
    #
    # @data_entry.setter
    # def data_entry(self, data_entry):
    #     """Sets the data_entry of this Dataset.
    #
    #
    #     :param data_entry: The data_entry of this Dataset.  # noqa: E501
    #     :type: list[DataEntry]
    #     """
    #
    #     self.data_entry = data_entry
    #
    # @property
    # def features(self):
    #     """Gets the features of this Dataset.  # noqa: E501
    #
    #
    #     :return: The features of this Dataset.  # noqa: E501
    #     :rtype: list[FeatureInfo]
    #     """
    #     return self.features
    #
    # @features.setter
    # def features(self, features):
    #     """Sets the features of this Dataset.
    #
    #
    #     :param features: The features of this Dataset.  # noqa: E501
    #     :type: list[FeatureInfo]
    #     """
    #
    #     self.features = features
    #
    # @property
    # def total_rows(self):
    #     """Gets the total_rows of this Dataset.  # noqa: E501
    #
    #
    #     :return: The total_rows of this Dataset.  # noqa: E501
    #     :rtype: int
    #     """
    #     return self.total_rows
    #
    # @total_rows.setter
    # def total_rows(self, total_rows):
    #     """Sets the total_rows of this Dataset.
    #
    #
    #     :param total_rows: The total_rows of this Dataset.  # noqa: E501
    #     :type: int
    #     """
    #
    #     self.total_rows = total_rows
    #
    # @property
    # def total_columns(self):
    #     """Gets the total_columns of this Dataset.  # noqa: E501
    #
    #
    #     :return: The total_columns of this Dataset.  # noqa: E501
    #     :rtype: int
    #     """
    #     return self.total_columns
    #
    # @total_columns.setter
    # def total_columns(self, total_columns):
    #     """Sets the total_columns of this Dataset.
    #
    #
    #     :param total_columns: The total_columns of this Dataset.  # noqa: E501
    #     :type: int
    #     """
    #
    #     self.total_columns = total_columns
    #
    # @property
    # def descriptors(self):
    #     """Gets the descriptors of this Dataset.  # noqa: E501
    #
    #
    #     :return: The descriptors of this Dataset.  # noqa: E501
    #     :rtype: list[str]
    #     """
    #     return self.descriptors
    #
    # @descriptors.setter
    # def descriptors(self, descriptors):
    #     """Sets the descriptors of this Dataset.
    #
    #
    #     :param descriptors: The descriptors of this Dataset.  # noqa: E501
    #     :type: list[str]
    #     """
    #     allowed_values = ["EXPERIMENTAL", "IMAGE", "GO", "MOPAC", "CDK", "PREDICTED"]  # noqa: E501
    #     if not set(descriptors).issubset(set(allowed_values)):
    #         raise ValueError(
    #             "Invalid values for `descriptors` [{0}], must be a subset of [{1}]"  # noqa: E501
    #             .format(", ".join(map(str, set(descriptors) - set(allowed_values))),  # noqa: E501
    #                     ", ".join(map(str, allowed_values)))
    #         )
    #
    #     self._descriptors = descriptors
    #
    # @property
    # def id(self):
    #     """Gets the id of this Dataset.  # noqa: E501
    #
    #
    #     :return: The id of this Dataset.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self.id
    #
    # @id.setter
    # def id(self, id):
    #     """Sets the id of this Dataset.
    #
    #
    #     :param id: The id of this Dataset.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self.id = id