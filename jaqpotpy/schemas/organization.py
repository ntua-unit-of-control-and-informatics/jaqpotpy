"""Author: Ioannis Pitoskas (jpitoskas@gmail.com)"""


class Organization:
    def __init__(self):
        pass

    def to_json(self):
        organization_dict = {}
        return organization_dict

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_json()})"
