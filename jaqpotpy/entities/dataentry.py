from .entryid import EntryId


class DataEntry(object):

    def __init__(self, entryId=None, values=None):

        self.entryId = None
        self.values = None

        if entryId is not None:
            self.entryId = entryId
        if values is not None:
            self.values = values

    # @property
    # def entry_id(self):
    #     """Gets the entry_id of this DataEntry.  # noqa: E501
    #
    #
    #     :return: The entry_id of this DataEntry.  # noqa: E501
    #     :rtype: EntryId
    #     """
    #     return self._entry_id
    #
    # @entry_id.setter
    # def entry_id(self, entry_id):
    #     """Sets the entry_id of this DataEntry.
    #
    #
    #     :param entry_id: The entry_id of this DataEntry.  # noqa: E501
    #     :type: EntryId
    #     """
    #
    #     self._entry_id = entry_id
    #
    # @property
    # def values(self):
    #     """Gets the values of this DataEntry.  # noqa: E501
    #
    #
    #     :return: The values of this DataEntry.  # noqa: E501
    #     :rtype: dict(str, object)
    #     """
    #     return self._values
    #
    # @values.setter
    # def values(self, values):
    #     """Sets the values of this DataEntry.
    #
    #
    #     :param values: The values of this DataEntry.  # noqa: E501
    #     :type: dict(str, object)
    #     """
    #
    #     self._values = values
