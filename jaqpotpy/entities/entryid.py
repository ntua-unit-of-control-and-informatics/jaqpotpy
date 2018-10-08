class EntryId(object):

    def __init__(self, name=None, ownerUUID=None, URI=None, type=None):
        self.name = None
        self.ownerUUID = None
        self.URI = None
        self.type = None

        if name is not None:
            self.name = name
        if ownerUUID is not None:
            self.owner_uuid = ownerUUID
        if URI is not None:
            self.URI = URI
        if type is not None:
            self.type = type

    # @property
    # def name(self):
    #     """Gets the name of this EntryId.  # noqa: E501
    #
    #
    #     :return: The name of this EntryId.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self._name
    #
    # @name.setter
    # def name(self, name):
    #     """Sets the name of this EntryId.
    #
    #
    #     :param name: The name of this EntryId.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self._name = name
    #
    # @property
    # def owner_uuid(self):
    #     """Gets the owner_uuid of this EntryId.  # noqa: E501
    #
    #
    #     :return: The owner_uuid of this EntryId.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self._owner_uuid
    #
    # @owner_uuid.setter
    # def owner_uuid(self, owner_uuid):
    #     """Sets the owner_uuid of this EntryId.
    #
    #
    #     :param owner_uuid: The owner_uuid of this EntryId.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self._owner_uuid = owner_uuid
    #
    # @property
    # def uri(self):
    #     """Gets the uri of this EntryId.  # noqa: E501
    #
    #
    #     :return: The uri of this EntryId.  # noqa: E501
    #     :rtype: str
    #     """
    #     return self._uri
    #
    # @uri.setter
    # def uri(self, uri):
    #     """Sets the uri of this EntryId.
    #
    #
    #     :param uri: The uri of this EntryId.  # noqa: E501
    #     :type: str
    #     """
    #
    #     self._uri = uri
