"""Author: Ioannis Pitoskas (jpitoskas@gmail.com)"""


class Library:
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version

    def to_json(self):
        library_dict = {
            "name": self.name,
            "version": self.version,
        }
        return library_dict

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_json()})"
