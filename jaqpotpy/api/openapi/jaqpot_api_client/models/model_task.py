from enum import Enum


class ModelTask(str, Enum):
    BINARY_CLASSIFICATION = "BINARY_CLASSIFICATION"
    MULTICLASS_CLASSIFICATION = "MULTICLASS_CLASSIFICATION"
    REGRESSION = "REGRESSION"

    def __str__(self) -> str:
        return str(self.value)
