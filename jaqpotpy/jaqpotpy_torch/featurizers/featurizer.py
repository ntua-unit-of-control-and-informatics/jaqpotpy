"""
Author: Ioannis Pitoskas
Contact: jpitoskas@gmail.com
"""

from abc import ABC, abstractmethod

class Featurizer(ABC):
    """
    Abstract base class for featurizers.
    """

    def __call__(self, *args, **kwargs):
        """
        Featurizes the input data.
        
        Returns:
            The featurized data.
        """
        return self.featurize(*args, **kwargs)
    
    @abstractmethod
    def featurize(self, *args, **kwargs):
        """
        Abstract method to featurize the input data.

        Returns:
            The featurized data.
        """
        pass

    