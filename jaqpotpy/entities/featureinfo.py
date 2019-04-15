class FeatureInfo(object):

    def __init__(self, name=None, units=None, conditions=None, category=None, ont=None, uri=None, key=None):

        self.name = None
        self.units = None
        self.conditions = None
        self.category = None
        self.key = None
        # self.ont = None
        self.uri = None

        if name is not None:
            self.name = name
        if units is not None:
            self.units = units
        if conditions is not None:
            self.conditions = conditions
        if category is not None:
            self.category = category
        if key is not None:
            self.key = key
        # if ont is not None:
        #     self.ont = ont
        if uri is not None:
            self.uri = uri

    # @property
    # def name(self):
    #     """Gets the name of this FeatureInfo.  # noqa: E501
    #
    #
    #     :return: The name of this FeatureInfo.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self.name
    #
    # @name.setter
    # def name(self, name):
    #     """Sets the name of this FeatureInfo.
    #
    #
    #     :param name: The name of this FeatureInfo.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self.name = name
    #
    # @property
    # def units(self):
    #     """Gets the units of this FeatureInfo.  # noqa: E501
    #
    #
    #     :return: The units of this FeatureInfo.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self.units
    #
    # @units.setter
    # def units(self, units):
    #     """Sets the units of this FeatureInfo.
    #
    #
    #     :param units: The units of this FeatureInfo.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self.units = units
    #
    # @property
    # def conditions(self):
    #     """Gets the conditions of this FeatureInfo.  # noqa: E501
    #
    #
    #     :return: The conditions of this FeatureInfo.  # noqa: E501
    #     :rtype: dict(str, object)
    #     """
    #     return self.conditions
    #
    # @conditions.setter
    # def conditions(self, conditions):
    #     """Sets the conditions of this FeatureInfo.
    #
    #
    #     :param conditions: The conditions of this FeatureInfo.  # noqa: E501
    #     :type: dict(str, object)
    #     """
    #
    #     self.conditions = conditions
    #
    # @property
    # def category(self):
    #     """Gets the category of this FeatureInfo.  # noqa: E501
    #
    #
    #     :return: The category of this FeatureInfo.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self.category
    #
    # @category.setter
    # def category(self, category):
    #     """Sets the category of this FeatureInfo.
    #
    #
    #     :param category: The category of this FeatureInfo.  # noqa: E501
    #     :type: str
    #     """
    #     allowedvalues = ["EXPERIMENTAL", "IMAGE", "GO", "MOPAC", "CDK", "PREDICTED"]
    #     if category not in allowedvalues:
    #         raise ValueError(
    #             "Invalid value for `category` ({0}), must be one of {1}"
    #             .format(category, allowedvalues)
    #         )
    #
    #     self.category = category
    #
    # @property
    # def ont(self):
    #     """Gets the ont of this FeatureInfo.  # noqa: E501
    #
    #
    #     :return: The ont of this FeatureInfo.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self.ont
    #
    # @ont.setter
    # def ont(self, ont):
    #     """Sets the ont of this FeatureInfo.
    #
    #
    #     :param ont: The ont of this FeatureInfo.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self.ont = ont
    #
    # @property
    # def uri(self):
    #     """Gets the uri of this FeatureInfo.  # noqa: E501
    #
    #
    #     :return: The uri of this FeatureInfo.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self.uri
    #
    # @uri.setter
    # def uri(self, uri):
    #     """Sets the uri of this FeatureInfo.
    #
    #
    #     :param uri: The uri of this FeatureInfo.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self.uri = uri
