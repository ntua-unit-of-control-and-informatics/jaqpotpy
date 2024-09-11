from enum import Enum


class FeatureFeatureDependency(str, Enum):
    DEPENDENT = "DEPENDENT"
    INDEPENDENT = "INDEPENDENT"

    def __str__(self) -> str:
        return str(self.value)
