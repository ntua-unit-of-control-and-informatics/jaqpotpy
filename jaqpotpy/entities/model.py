class PretrainedModel(object):

    def __init__(self, raw_model=None, pmml_model=None, additional_info=None, dependent_features=None, independent_features=None, predicted_features=None, implemented_in=None, implemented_with=None, title=None, discription=None, algorithm=None):

        self._raw_model = None
        self._pmml_model = None
        self._additional_info = None
        self._dependent_features = None
        self._independent_features = None
        self._predicted_features = None
        self._implemented_in = None
        self._implemented_with = None
        self._title = None
        self._discription = None
        self._algorithm = None
        self.discriminator = None

        if raw_model is not None:
            self.raw_model = raw_model
        if pmml_model is not None:
            self.pmml_model = pmml_model
        if additional_info is not None:
            self.additional_info = additional_info
        if dependent_features is not None:
            self.dependent_features = dependent_features
        if independent_features is not None:
            self.independent_features = independent_features
        if predicted_features is not None:
            self.predicted_features = predicted_features
        if implemented_in is not None:
            self.implemented_in = implemented_in
        if implemented_with is not None:
            self.implemented_with = implemented_with
        if title is not None:
            self.title = title
        if discription is not None:
            self.discription = discription
        if algorithm is not None:
            self.algorithm = algorithm

    @property
    def raw_model(self):
        """Gets the raw_model of this PretrainedModel.  # noqa: E501


        :return: The raw_model of this PretrainedModel.  # noqa: E501
        :rtype: object
        """
        return self._raw_model

    @raw_model.setter
    def raw_model(self, raw_model):
        """Sets the raw_model of this PretrainedModel.


        :param raw_model: The raw_model of this PretrainedModel.  # noqa: E501
        :type: object
        """

        self._raw_model = raw_model

    @property
    def pmml_model(self):
        """Gets the pmml_model of this PretrainedModel.  # noqa: E501


        :return: The pmml_model of this PretrainedModel.  # noqa: E501
        :rtype: object
        """
        return self._pmml_model

    @pmml_model.setter
    def pmml_model(self, pmml_model):
        """Sets the pmml_model of this PretrainedModel.


        :param pmml_model: The pmml_model of this PretrainedModel.  # noqa: E501
        :type: object
        """

        self._pmml_model = pmml_model

    @property
    def additional_info(self):
        """Gets the additional_info of this PretrainedModel.  # noqa: E501


        :return: The additional_info of this PretrainedModel.  # noqa: E501
        :rtype: object
        """
        return self._additional_info

    @additional_info.setter
    def additional_info(self, additional_info):
        """Sets the additional_info of this PretrainedModel.


        :param additional_info: The additional_info of this PretrainedModel.  # noqa: E501
        :type: object
        """

        self._additional_info = additional_info

    @property
    def dependent_features(self):
        """Gets the dependent_features of this PretrainedModel.  # noqa: E501


        :return: The dependent_features of this PretrainedModel.  # noqa: E501
        :rtype: list[str]
        """
        return self._dependent_features

    @dependent_features.setter
    def dependent_features(self, dependent_features):
        """Sets the dependent_features of this PretrainedModel.


        :param dependent_features: The dependent_features of this PretrainedModel.  # noqa: E501
        :type: list[str]
        """

        self._dependent_features = dependent_features

    @property
    def independent_features(self):
        """Gets the independent_features of this PretrainedModel.  # noqa: E501


        :return: The independent_features of this PretrainedModel.  # noqa: E501
        :rtype: list[str]
        """
        return self._independent_features

    @independent_features.setter
    def independent_features(self, independent_features):
        """Sets the independent_features of this PretrainedModel.


        :param independent_features: The independent_features of this PretrainedModel.  # noqa: E501
        :type: list[str]
        """

        self._independent_features = independent_features

    @property
    def predicted_features(self):
        """Gets the predicted_features of this PretrainedModel.  # noqa: E501


        :return: The predicted_features of this PretrainedModel.  # noqa: E501
        :rtype: list[str]
        """
        return self._predicted_features

    @predicted_features.setter
    def predicted_features(self, predicted_features):
        """Sets the predicted_features of this PretrainedModel.


        :param predicted_features: The predicted_features of this PretrainedModel.  # noqa: E501
        :type: list[str]
        """

        self._predicted_features = predicted_features

    @property
    def implemented_in(self):
        """Gets the implemented_in of this PretrainedModel.  # noqa: E501


        :return: The implemented_in of this PretrainedModel.  # noqa: E501
        :rtype: list[str]
        """
        return self._implemented_in

    @implemented_in.setter
    def implemented_in(self, implemented_in):
        """Sets the implemented_in of this PretrainedModel.


        :param implemented_in: The implemented_in of this PretrainedModel.  # noqa: E501
        :type: list[str]
        """

        self._implemented_in = implemented_in

    @property
    def implemented_with(self):
        """Gets the implemented_with of this PretrainedModel.  # noqa: E501


        :return: The implemented_with of this PretrainedModel.  # noqa: E501
        :rtype: list[str]
        """
        return self._implemented_with

    @implemented_with.setter
    def implemented_with(self, implemented_with):
        """Sets the implemented_with of this PretrainedModel.


        :param implemented_with: The implemented_with of this PretrainedModel.  # noqa: E501
        :type: list[str]
        """

        self._implemented_with = implemented_with

    @property
    def title(self):
        """Gets the title of this PretrainedModel.  # noqa: E501


        :return: The title of this PretrainedModel.  # noqa: E501
        :rtype: list[str]
        """
        return self._title

    @title.setter
    def title(self, title):
        """Sets the title of this PretrainedModel.


        :param title: The title of this PretrainedModel.  # noqa: E501
        :type: list[str]
        """

        self._title = title

    @property
    def discription(self):
        """Gets the discription of this PretrainedModel.  # noqa: E501


        :return: The discription of this PretrainedModel.  # noqa: E501
        :rtype: list[str]
        """
        return self._discription

    @discription.setter
    def discription(self, discription):
        """Sets the discription of this PretrainedModel.


        :param discription: The discription of this PretrainedModel.  # noqa: E501
        :type: list[str]
        """

        self._discription = discription

    @property
    def algorithm(self):
        """Gets the algorithm of this PretrainedModel.  # noqa: E501


        :return: The algorithm of this PretrainedModel.  # noqa: E501
        :rtype: list[str]
        """
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        """Sets the algorithm of this PretrainedModel.


        :param algorithm: The algorithm of this PretrainedModel.  # noqa: E501
        :type: list[str]
        """

        self._algorithm = algorithm