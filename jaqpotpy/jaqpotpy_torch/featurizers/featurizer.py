from abc import ABC, abstractmethod

class Featurizer(ABC):
    @abstractmethod
    def featurize(self, *args, **kwargs):
        pass