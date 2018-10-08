from .meta import MetaInfo  # noqa: F401,E501


class Feature(object):

    def __init__(self, meta=None, ontological_classes=None, visible=False, temporary=False, featured=False, units=None, predictor_for=None, admissible_values=None, actual_independent_feature_name=None, from_pretrained=False, id=None):

        self.meta = None
        self.ontologicalClasses = None
        self.visible = None
        self.temporary = None
        self.featured = None
        self.units = None
        self.predictorFor = None
        self.admissibleValues = None
        self.actualIndependentFeatureName = None
        self.fromPretrained = None
        self.id = None
        self.discriminator = None


        # if meta is not None:
        #     self.meta = meta
        # if ontological_classes is not None:
        #     self.ontological_classes = ontological_classes
        # if visible is not None:
        #     self.visible = visible
        # if temporary is not None:
        #     self.temporary = temporary
        # if featured is not None:
        #     self.featured = featured
        # if units is not None:
        #     self.units = units
        # if predictor_for is not None:
        #     self.predictor_for = predictor_for
        # if admissible_values is not None:
        #     self.admissible_values = admissible_values
        # if actual_independent_feature_name is not None:
        #     self.actual_independent_feature_name = actual_independent_feature_name
        # if from_pretrained is not None:
        #     self.from_pretrained = from_pretrained
        # if id is not None:
        #     self.id = id

    # @property
    # def meta(self):
    #     """Gets the meta of this Feature.  # noqa: E501
    #
    #
    #     :return: The meta of this Feature.  # noqa: E501
    #     :rtype: MetaInfo
    #     """
    #     return self._meta
    #
    # @meta.setter
    # def meta(self, meta):
    #     """Sets the meta of this Feature.
    #
    #
    #     :param meta: The meta of this Feature.  # noqa: E501
    #     :type: MetaInfo
    #     """
    #
    #     self._meta = meta
    #
    # @property
    # def ontological_classes(self):
    #     """Gets the ontological_classes of this Feature.  # noqa: E501
    #
    #
    #     :return: The ontological_classes of this Feature.  # noqa: E501
    #     :rtype: list[str]
    #     """
    #     return self._ontological_classes
    #
    # @ontological_classes.setter
    # def ontological_classes(self, ontological_classes):
    #     """Sets the ontological_classes of this Feature.
    #
    #
    #     :param ontological_classes: The ontological_classes of this Feature.  # noqa: E501
    #     :type: list[str]
    #     """
    #
    #     self._ontological_classes = ontological_classes
    #
    # @property
    # def visible(self):
    #     """Gets the visible of this Feature.  # noqa: E501
    #
    #
    #     :return: The visible of this Feature.  # noqa: E501
    #     :rtype: bool
    #     """
    #     return self._visible
    #
    # @visible.setter
    # def visible(self, visible):
    #     """Sets the visible of this Feature.
    #
    #
    #     :param visible: The visible of this Feature.  # noqa: E501
    #     :type: bool
    #     """
    #
    #     self._visible = visible
    #
    # @property
    # def temporary(self):
    #     """Gets the temporary of this Feature.  # noqa: E501
    #
    #
    #     :return: The temporary of this Feature.  # noqa: E501
    #     :rtype: bool
    #     """
    #     return self._temporary
    #
    # @temporary.setter
    # def temporary(self, temporary):
    #     """Sets the temporary of this Feature.
    #
    #
    #     :param temporary: The temporary of this Feature.  # noqa: E501
    #     :type: bool
    #     """
    #
    #     self._temporary = temporary
    #
    # @property
    # def featured(self):
    #     """Gets the featured of this Feature.  # noqa: E501
    #
    #
    #     :return: The featured of this Feature.  # noqa: E501
    #     :rtype: bool
    #     """
    #     return self._featured
    #
    # @featured.setter
    # def featured(self, featured):
    #     """Sets the featured of this Feature.
    #
    #
    #     :param featured: The featured of this Feature.  # noqa: E501
    #     :type: bool
    #     """
    #
    #     self._featured = featured
    #
    # @property
    # def units(self):
    #     """Gets the units of this Feature.  # noqa: E501
    #
    #
    #     :return: The units of this Feature.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self._units
    #
    # @units.setter
    # def units(self, units):
    #     """Sets the units of this Feature.
    #
    #
    #     :param units: The units of this Feature.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self._units = units
    #
    # @property
    # def predictor_for(self):
    #     """Gets the predictor_for of this Feature.  # noqa: E501
    #
    #
    #     :return: The predictor_for of this Feature.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self._predictor_for
    #
    # @predictor_for.setter
    # def predictor_for(self, predictor_for):
    #     """Sets the predictor_for of this Feature.
    #
    #
    #     :param predictor_for: The predictor_for of this Feature.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self._predictor_for = predictor_for
    #
    # @property
    # def admissible_values(self):
    #     """Gets the admissible_values of this Feature.  # noqa: E501
    #
    #
    #     :return: The admissible_values of this Feature.  # noqa: E501
    #     :rtype: list[str]
    #     """
    #     return self._admissible_values
    #
    # @admissible_values.setter
    # def admissible_values(self, admissible_values):
    #     """Sets the admissible_values of this Feature.
    #
    #
    #     :param admissible_values: The admissible_values of this Feature.  # noqa: E501
    #     :type: list[str]
    #     """
    #
    #     self._admissible_values = admissible_values
    #
    # @property
    # def actual_independent_feature_name(self):
    #     """Gets the actual_independent_feature_name of this Feature.  # noqa: E501
    #
    #
    #     :return: The actual_independent_feature_name of this Feature.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self._actual_independent_feature_name
    #
    # @actual_independent_feature_name.setter
    # def actual_independent_feature_name(self, actual_independent_feature_name):
    #     """Sets the actual_independent_feature_name of this Feature.
    #
    #
    #     :param actual_independent_feature_name: The actual_independent_feature_name of this Feature.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self._actual_independent_feature_name = actual_independent_feature_name
    #
    # @property
    # def from_pretrained(self):
    #     """Gets the from_pretrained of this Feature.  # noqa: E501
    #
    #
    #     :return: The from_pretrained of this Feature.  # noqa: E501
    #     :rtype: bool
    #     """
    #     return self._from_pretrained
    #
    # @from_pretrained.setter
    # def from_pretrained(self, from_pretrained):
    #     """Sets the from_pretrained of this Feature.
    #
    #
    #     :param from_pretrained: The from_pretrained of this Feature.  # noqa: E501
    #     :type: bool
    #     """
    #
    #     self._from_pretrained = from_pretrained
    #
    # @property
    # def id(self):
    #     """Gets the id of this Feature.  # noqa: E501
    #
    #
    #     :return: The id of this Feature.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self._id
    #
    # @id.setter
    # def id(self, id):
    #     """Sets the id of this Feature.
    #
    #
    #     :param id: The id of this Feature.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self._id = id
