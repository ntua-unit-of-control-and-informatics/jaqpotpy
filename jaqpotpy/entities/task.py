from .error_report import ErrorReport  # noqa: F401,E501
from .meta import MetaInfo  # noqa: F401,E501


class Task(object):

    def __init__(self, meta=None, ontological_classes=None, visible=False, temporary=False, featured=False, result_uri=None, result=None, percentage_completed=None, error_report=None, http_status=None, duration=None, type=None, id=None, status=None):  # noqa: E501
        """Task - a model defined in Swagger"""  # noqa: E501

        self._meta = None
        self._ontological_classes = None
        self._visible = None
        self._temporary = None
        self._featured = None
        self._result_uri = None
        self._result = None
        self._percentage_completed = None
        self._error_report = None
        self._http_status = None
        self._duration = None
        self._type = None
        self._id = None
        self._status = None
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
        if result_uri is not None:
            self.result_uri = result_uri
        if result is not None:
            self.result = result
        if percentage_completed is not None:
            self.percentage_completed = percentage_completed
        if error_report is not None:
            self.error_report = error_report
        if http_status is not None:
            self.http_status = http_status
        if duration is not None:
            self.duration = duration
        if type is not None:
            self.type = type
        if id is not None:
            self.id = id
        if status is not None:
            self.status = status

    @property
    def meta(self):
        """Gets the meta of this Task.  # noqa: E501


        :return: The meta of this Task.  # noqa: E501
        :rtype: MetaInfo
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        """Sets the meta of this Task.


        :param meta: The meta of this Task.  # noqa: E501
        :type: MetaInfo
        """

        self._meta = meta

    @property
    def ontological_classes(self):
        """Gets the ontological_classes of this Task.  # noqa: E501


        :return: The ontological_classes of this Task.  # noqa: E501
        :rtype: list[str]
        """
        return self._ontological_classes

    @ontological_classes.setter
    def ontological_classes(self, ontological_classes):
        """Sets the ontological_classes of this Task.


        :param ontological_classes: The ontological_classes of this Task.  # noqa: E501
        :type: list[str]
        """

        self._ontological_classes = ontological_classes

    @property
    def visible(self):
        """Gets the visible of this Task.  # noqa: E501


        :return: The visible of this Task.  # noqa: E501
        :rtype: bool
        """
        return self._visible

    @visible.setter
    def visible(self, visible):
        """Sets the visible of this Task.


        :param visible: The visible of this Task.  # noqa: E501
        :type: bool
        """

        self._visible = visible

    @property
    def temporary(self):
        """Gets the temporary of this Task.  # noqa: E501


        :return: The temporary of this Task.  # noqa: E501
        :rtype: bool
        """
        return self._temporary

    @temporary.setter
    def temporary(self, temporary):
        """Sets the temporary of this Task.


        :param temporary: The temporary of this Task.  # noqa: E501
        :type: bool
        """

        self._temporary = temporary

    @property
    def featured(self):
        """Gets the featured of this Task.  # noqa: E501


        :return: The featured of this Task.  # noqa: E501
        :rtype: bool
        """
        return self._featured

    @featured.setter
    def featured(self, featured):
        """Sets the featured of this Task.


        :param featured: The featured of this Task.  # noqa: E501
        :type: bool
        """

        self._featured = featured

    @property
    def result_uri(self):
        """Gets the result_uri of this Task.  # noqa: E501


        :return: The result_uri of this Task.  # noqa: E501
        :rtype: str
        """
        return self._result_uri

    @result_uri.setter
    def result_uri(self, result_uri):
        """Sets the result_uri of this Task.


        :param result_uri: The result_uri of this Task.  # noqa: E501
        :type: str
        """

        self._result_uri = result_uri

    @property
    def result(self):
        """Gets the result of this Task.  # noqa: E501


        :return: The result of this Task.  # noqa: E501
        :rtype: str
        """
        return self._result

    @result.setter
    def result(self, result):
        """Sets the result of this Task.


        :param result: The result of this Task.  # noqa: E501
        :type: str
        """

        self._result = result

    @property
    def percentage_completed(self):
        """Gets the percentage_completed of this Task.  # noqa: E501


        :return: The percentage_completed of this Task.  # noqa: E501
        :rtype: float
        """
        return self._percentage_completed

    @percentage_completed.setter
    def percentage_completed(self, percentage_completed):
        """Sets the percentage_completed of this Task.


        :param percentage_completed: The percentage_completed of this Task.  # noqa: E501
        :type: float
        """

        self._percentage_completed = percentage_completed

    @property
    def error_report(self):
        """Gets the error_report of this Task.  # noqa: E501


        :return: The error_report of this Task.  # noqa: E501
        :rtype: ErrorReport
        """
        return self._error_report

    @error_report.setter
    def error_report(self, error_report):
        """Sets the error_report of this Task.


        :param error_report: The error_report of this Task.  # noqa: E501
        :type: ErrorReport
        """

        self._error_report = error_report

    @property
    def http_status(self):
        """Gets the http_status of this Task.  # noqa: E501


        :return: The http_status of this Task.  # noqa: E501
        :rtype: int
        """
        return self._http_status

    @http_status.setter
    def http_status(self, http_status):
        """Sets the http_status of this Task.


        :param http_status: The http_status of this Task.  # noqa: E501
        :type: int
        """

        self._http_status = http_status

    @property
    def duration(self):
        """Gets the duration of this Task.  # noqa: E501


        :return: The duration of this Task.  # noqa: E501
        :rtype: int
        """
        return self._duration

    @duration.setter
    def duration(self, duration):
        """Sets the duration of this Task.


        :param duration: The duration of this Task.  # noqa: E501
        :type: int
        """

        self._duration = duration

    @property
    def type(self):
        """Gets the type of this Task.  # noqa: E501


        :return: The type of this Task.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this Task.


        :param type: The type of this Task.  # noqa: E501
        :type: str
        """
        allowed_values = ["TRAINING", "PREDICTION", "PREPARATION", "VALIDATION"]  # noqa: E501
        if type not in allowed_values:
            raise ValueError(
                "Invalid value for `type` ({0}), must be one of {1}"  # noqa: E501
                .format(type, allowed_values)
            )

        self._type = type

    @property
    def id(self):
        """Gets the id of this Task.  # noqa: E501


        :return: The id of this Task.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this Task.


        :param id: The id of this Task.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def status(self):
        """Gets the status of this Task.  # noqa: E501


        :return: The status of this Task.  # noqa: E501
        :rtype: str
        """
        return self._status

    @status.setter
    def status(self, status):
        """Sets the status of this Task.


        :param status: The status of this Task.  # noqa: E501
        :type: str
        """
        allowed_values = ["RUNNING", "COMPLETED", "CANCELLED", "ERROR", "REJECTED", "QUEUED"]  # noqa: E501
        if status not in allowed_values:
            raise ValueError(
                "Invalid value for `status` ({0}), must be one of {1}"  # noqa: E501
                .format(status, allowed_values)
            )

        self._status = status
