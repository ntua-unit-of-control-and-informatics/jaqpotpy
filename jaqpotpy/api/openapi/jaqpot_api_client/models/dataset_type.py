from enum import Enum


class DatasetType(str, Enum):
    PREDICTION = "PREDICTION"

    def __str__(self) -> str:
        return str(self.value)
