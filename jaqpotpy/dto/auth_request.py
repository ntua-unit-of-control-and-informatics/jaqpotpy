class AuthRequest:

    def __init__(self, username='', authtoken=''):
        self.userName = username
        self.authToken = authtoken

    # @property
    # def userName(self):
    #     return self._userName
    #
    # @property
    # def authToken(self):
    #     return self._authToken
    #
    # @userName.setter
    # def userName(self, value):
    #     self.userName = value
    #
    # @authToken.setter
    # def authToken(self, value):
    #     self.authToken = value
    #
    # @userName.deleter
    # def userName(self):
    #     del self.userName
    #
    # @authToken.deleter
    # def authToken(self):
    #     del self.authToken
