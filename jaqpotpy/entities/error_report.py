from .meta import MetaInfo  # noqa: F401,E501


class ErrorReport(object):

    def __init__(self, meta=None, ontological_classes=None, visible=False, temporary=False, featured=False, code=None, actor=None, message=None, details=None, http_status=None, id=None):

        self._meta = None
        self._ontological_classes = None
        self._visible = None
        self._temporary = None
        self._featured = None
        self._code = None
        self._actor = None
        self._message = None
        self._details = None
        self._http_status = None
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
        if code is not None:
            self.code = code
        if actor is not None:
            self.actor = actor
        if message is not None:
            self.message = message
        if details is not None:
            self.details = details
        if http_status is not None:
            self.http_status = http_status
        if id is not None:
            self.id = id

    @property
    def meta(self):
        """Gets the meta of this ErrorReport.  # noqa: E501


        :return: The meta of this ErrorReport.  # noqa: E501
        :rtype: MetaInfo
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        """Sets the meta of this ErrorReport.


        :param meta: The meta of this ErrorReport.  # noqa: E501
        :type: MetaInfo
        """

        self._meta = meta

    @property
    def ontological_classes(self):
        """Gets the ontological_classes of this ErrorReport.  # noqa: E501


        :return: The ontological_classes of this ErrorReport.  # noqa: E501
        :rtype: list[str]
        """
        return self._ontological_classes

    @ontological_classes.setter
    def ontological_classes(self, ontological_classes):
        """Sets the ontological_classes of this ErrorReport.


        :param ontological_classes: The ontological_classes of this ErrorReport.  # noqa: E501
        :type: list[str]
        """

        self._ontological_classes = ontological_classes

    @property
    def visible(self):
        """Gets the visible of this ErrorReport.  # noqa: E501


        :return: The visible of this ErrorReport.  # noqa: E501
        :rtype: bool
        """
        return self._visible

    @visible.setter
    def visible(self, visible):
        """Sets the visible of this ErrorReport.


        :param visible: The visible of this ErrorReport.  # noqa: E501
        :type: bool
        """

        self._visible = visible

    @property
    def temporary(self):
        """Gets the temporary of this ErrorReport.  # noqa: E501


        :return: The temporary of this ErrorReport.  # noqa: E501
        :rtype: bool
        """
        return self._temporary

    @temporary.setter
    def temporary(self, temporary):
        """Sets the temporary of this ErrorReport.


        :param temporary: The temporary of this ErrorReport.  # noqa: E501
        :type: bool
        """

        self._temporary = temporary

    @property
    def featured(self):
        """Gets the featured of this ErrorReport.  # noqa: E501


        :return: The featured of this ErrorReport.  # noqa: E501
        :rtype: bool
        """
        return self._featured

    @featured.setter
    def featured(self, featured):
        """Sets the featured of this ErrorReport.


        :param featured: The featured of this ErrorReport.  # noqa: E501
        :type: bool
        """

        self._featured = featured

    @property
    def code(self):
        """Gets the code of this ErrorReport.  # noqa: E501

        Error code  # noqa: E501

        :return: The code of this ErrorReport.  # noqa: E501
        :rtype: str
        """
        return self._code

    @code.setter
    def code(self, code):
        """Sets the code of this ErrorReport.

        Error code  # noqa: E501

        :param code: The code of this ErrorReport.  # noqa: E501
        :type: str
        """

        self._code = code

    @property
    def actor(self):
        """Gets the actor of this ErrorReport.  # noqa: E501

        Who is to blame  # noqa: E501

        :return: The actor of this ErrorReport.  # noqa: E501
        :rtype: str
        """
        return self._actor

    @actor.setter
    def actor(self, actor):
        """Sets the actor of this ErrorReport.

        Who is to blame  # noqa: E501

        :param actor: The actor of this ErrorReport.  # noqa: E501
        :type: str
        """

        self._actor = actor

    @property
    def message(self):
        """Gets the message of this ErrorReport.  # noqa: E501

        Short error message  # noqa: E501

        :return: The message of this ErrorReport.  # noqa: E501
        :rtype: str
        """
        return self._message

    @message.setter
    def message(self, message):
        """Sets the message of this ErrorReport.

        Short error message  # noqa: E501

        :param message: The message of this ErrorReport.  # noqa: E501
        :type: str
        """

        self._message = message

    @property
    def details(self):
        """Gets the details of this ErrorReport.  # noqa: E501

        Details to be used for debugging.  # noqa: E501

        :return: The details of this ErrorReport.  # noqa: E501
        :rtype: str
        """
        return self._details

    @details.setter
    def details(self, details):
        """Sets the details of this ErrorReport.

        Details to be used for debugging.  # noqa: E501

        :param details: The details of this ErrorReport.  # noqa: E501
        :type: str
        """

        self._details = details

    @property
    def http_status(self):
        """Gets the http_status of this ErrorReport.  # noqa: E501

        Accompanying HTTP status.  # noqa: E501

        :return: The http_status of this ErrorReport.  # noqa: E501
        :rtype: int
        """
        return self._http_status

    @http_status.setter
    def http_status(self, http_status):
        """Sets the http_status of this ErrorReport.

        Accompanying HTTP status.  # noqa: E501

        :param http_status: The http_status of this ErrorReport.  # noqa: E501
        :type: int
        """

        self._http_status = http_status

    @property
    def id(self):
        """Gets the id of this ErrorReport.  # noqa: E501


        :return: The id of this ErrorReport.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this ErrorReport.


        :param id: The id of this ErrorReport.  # noqa: E501
        :type: str
        """

        self._id = id
