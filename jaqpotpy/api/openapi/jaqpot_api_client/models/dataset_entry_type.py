from enum import Enum


class DatasetEntryType(str, Enum):
    ARRAY = "ARRAY"

    def __str__(self) -> str:
        return str(self.value)
