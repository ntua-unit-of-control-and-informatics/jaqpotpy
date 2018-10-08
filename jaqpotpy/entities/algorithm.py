from .meta import MetaInfo
from .parameter import Parameter


class Algorithm(object):

    def __init__(self, meta=None, ontological_classes=None, visible=False, temporary=False,
                 featured=False, parameters=None, ranking=None, bibtex=None, training_service=None,
                 prediction_service=None, report_service=None, _id=None):
        self._meta = None
        self._ontological_classes = None
        self._visible = None
        self._temporary = None
        self._featured = None
        self._parameters = None
        self._ranking = None
        self._bibtex = None
        self._training_service = None
        self._prediction_service = None
        self._report_service = None
        self._id = None
        self.discriminator = None

        if meta is not None:
            self.meta = meta
        if ontological_classes is not None:
            self.ontological_classes = ontological_classes
        if visible is not None:
            self.visible = visible
        if temporary is not None:
            self.temporary = temporary
        if featured is not None:
            self.featured = featured
        if parameters is not None:
            self.parameters = parameters
        if ranking is not None:
            self.ranking = ranking
        if bibtex is not None:
            self.bibtex = bibtex
        if training_service is not None:
            self.training_service = training_service
        if prediction_service is not None:
            self.prediction_service = prediction_service
        if report_service is not None:
            self.report_service = report_service
        if _id is not None:
            self._id = _id

    # @property
    # def meta(self):
    #     """Gets the meta of this Algorithm.  # noqa: E501
    #
    #
    #     :return: The meta of this Algorithm.  # noqa: E501
    #     :rtype: MetaInfo
    #     """
    #     return self._meta
    #
    # @meta.setter
    # def meta(self, meta):
    #     """Sets the meta of this Algorithm.
    #
    #
    #     :param meta: The meta of this Algorithm.  # noqa: E501
    #     :type: MetaInfo
    #     """
    #
    #     self._meta = meta
    #
    # @property
    # def ontological_classes(self):
    #     """Gets the ontological_classes of this Algorithm.  # noqa: E501
    #
    #
    #     :return: The ontological_classes of this Algorithm.  # noqa: E501
    #     :rtype: list[str]
    #     """
    #     return self._ontological_classes
    #
    # @ontological_classes.setter
    # def ontological_classes(self, ontological_classes):
    #     """Sets the ontological_classes of this Algorithm.
    #
    #
    #     :param ontological_classes: The ontological_classes of this Algorithm.  # noqa: E501
    #     :type: list[str]
    #     """
    #
    #     self._ontological_classes = ontological_classes
    #
    # @property
    # def visible(self):
    #     """Gets the visible of this Algorithm.  # noqa: E501
    #
    #
    #     :return: The visible of this Algorithm.  # noqa: E501
    #     :rtype: bool
    #     """
    #     return self._visible
    #
    # @visible.setter
    # def visible(self, visible):
    #     """Sets the visible of this Algorithm.
    #
    #
    #     :param visible: The visible of this Algorithm.  # noqa: E501
    #     :type: bool
    #     """
    #
    #     self._visible = visible
    #
    # @property
    # def temporary(self):
    #     """Gets the temporary of this Algorithm.  # noqa: E501
    #
    #
    #     :return: The temporary of this Algorithm.  # noqa: E501
    #     :rtype: bool
    #     """
    #     return self._temporary
    #
    # @temporary.setter
    # def temporary(self, temporary):
    #     """Sets the temporary of this Algorithm.
    #
    #
    #     :param temporary: The temporary of this Algorithm.  # noqa: E501
    #     :type: bool
    #     """
    #
    #     self._temporary = temporary
    #
    # @property
    # def featured(self):
    #     """Gets the featured of this Algorithm.  # noqa: E501
    #
    #
    #     :return: The featured of this Algorithm.  # noqa: E501
    #     :rtype: bool
    #     """
    #     return self._featured
    #
    # @featured.setter
    # def featured(self, featured):
    #     """Sets the featured of this Algorithm.
    #
    #
    #     :param featured: The featured of this Algorithm.  # noqa: E501
    #     :type: bool
    #     """
    #
    #     self._featured = featured
    #
    # @property
    # def parameters(self):
    #     """Gets the parameters of this Algorithm.  # noqa: E501
    #
    #
    #     :return: The parameters of this Algorithm.  # noqa: E501
    #     :rtype: list[Parameter]
    #     """
    #     return self._parameters
    #
    # @parameters.setter
    # def parameters(self, parameters):
    #     """Sets the parameters of this Algorithm.
    #
    #
    #     :param parameters: The parameters of this Algorithm.  # noqa: E501
    #     :type: list[Parameter]
    #     """
    #
    #     self._parameters = parameters
    #
    # @property
    # def ranking(self):
    #     """Gets the ranking of this Algorithm.  # noqa: E501
    #
    #
    #     :return: The ranking of this Algorithm.  # noqa: E501
    #     :rtype: int
    #     """
    #     return self._ranking
    #
    # @ranking.setter
    # def ranking(self, ranking):
    #     """Sets the ranking of this Algorithm.
    #
    #
    #     :param ranking: The ranking of this Algorithm.  # noqa: E501
    #     :type: int
    #     """
    #
    #     self._ranking = ranking
    #
    # @property
    # def bibtex(self):
    #     """Gets the bibtex of this Algorithm.  # noqa: E501
    #
    #
    #     :return: The bibtex of this Algorithm.  # noqa: E501
    #     :rtype: list[BibTeX]
    #     """
    #     return self._bibtex
    #
    # @bibtex.setter
    # def bibtex(self, bibtex):
    #     """Sets the bibtex of this Algorithm.
    #
    #
    #     :param bibtex: The bibtex of this Algorithm.  # noqa: E501
    #     :type: list[BibTeX]
    #     """
    #
    #     self._bibtex = bibtex
    #
    # @property
    # def training_service(self):
    #     """Gets the training_service of this Algorithm.  # noqa: E501
    #
    #
    #     :return: The training_service of this Algorithm.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self._training_service
    #
    # @training_service.setter
    # def training_service(self, training_service):
    #     """Sets the training_service of this Algorithm.
    #
    #
    #     :param training_service: The training_service of this Algorithm.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self._training_service = training_service
    #
    # @property
    # def prediction_service(self):
    #     """Gets the prediction_service of this Algorithm.  # noqa: E501
    #
    #
    #     :return: The prediction_service of this Algorithm.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self._prediction_service
    #
    # @prediction_service.setter
    # def prediction_service(self, prediction_service):
    #     """Sets the prediction_service of this Algorithm.
    #
    #
    #     :param prediction_service: The prediction_service of this Algorithm.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self._prediction_service = prediction_service
    #
    # @property
    # def report_service(self):
    #     """Gets the report_service of this Algorithm.  # noqa: E501
    #
    #
    #     :return: The report_service of this Algorithm.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self._report_service
    #
    # @report_service.setter
    # def report_service(self, report_service):
    #     """Sets the report_service of this Algorithm.
    #
    #
    #     :param report_service: The report_service of this Algorithm.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self._report_service = report_service
    #
    # @property
    # def _id(self):
    #     """Gets the id of this Algorithm.  # noqa: E501
    #
    #
    #     :return: The id of this Algorithm.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self._id
    #
    # @_id.setter
    # def _id(self, _id):
    #     """Sets the id of this Algorithm.
    #
    #
    #     :param _id: The id of this Algorithm.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self._id = _id
