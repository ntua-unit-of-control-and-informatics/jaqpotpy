class Parameter(object):

    def __init__(self, required=False, description=None, vendor_extensions=None, pattern=None, _in=None, name=None):

        self._required = None
        self._description = None
        self._vendor_extensions = None
        self._pattern = None
        self.__in = None
        self._name = None
        self.discriminator = None

        if required is not None:
            self.required = required
        if description is not None:
            self.description = description
        if vendor_extensions is not None:
            self.vendor_extensions = vendor_extensions
        if pattern is not None:
            self.pattern = pattern
        if _in is not None:
            self._in = _in
        if name is not None:
            self.name = name

    @property
    def required(self):
        """Gets the required of this Parameter.  # noqa: E501


        :return: The required of this Parameter.  # noqa: E501
        :rtype: bool
        """
        return self._required

    @required.setter
    def required(self, required):
        """Sets the required of this Parameter.


        :param required: The required of this Parameter.  # noqa: E501
        :type: bool
        """

        self._required = required

    @property
    def description(self):
        """Gets the description of this Parameter.  # noqa: E501


        :return: The description of this Parameter.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """Sets the description of this Parameter.


        :param description: The description of this Parameter.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def vendor_extensions(self):
        """Gets the vendor_extensions of this Parameter.  # noqa: E501


        :return: The vendor_extensions of this Parameter.  # noqa: E501
        :rtype: dict(str, object)
        """
        return self._vendor_extensions

    @vendor_extensions.setter
    def vendor_extensions(self, vendor_extensions):
        """Sets the vendor_extensions of this Parameter.


        :param vendor_extensions: The vendor_extensions of this Parameter.  # noqa: E501
        :type: dict(str, object)
        """

        self._vendor_extensions = vendor_extensions

    @property
    def pattern(self):
        """Gets the pattern of this Parameter.  # noqa: E501


        :return: The pattern of this Parameter.  # noqa: E501
        :rtype: str
        """
        return self._pattern

    @pattern.setter
    def pattern(self, pattern):
        """Sets the pattern of this Parameter.


        :param pattern: The pattern of this Parameter.  # noqa: E501
        :type: str
        """

        self._pattern = pattern

    @property
    def _in(self):
        """Gets the _in of this Parameter.  # noqa: E501


        :return: The _in of this Parameter.  # noqa: E501
        :rtype: str
        """
        return self.__in

    @_in.setter
    def _in(self, _in):
        """Sets the _in of this Parameter.


        :param _in: The _in of this Parameter.  # noqa: E501
        :type: str
        """

        self.__in = _in

    @property
    def name(self):
        """Gets the name of this Parameter.  # noqa: E501


        :return: The name of this Parameter.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this Parameter.


        :param name: The name of this Parameter.  # noqa: E501
        :type: str
        """

        self._name = name
