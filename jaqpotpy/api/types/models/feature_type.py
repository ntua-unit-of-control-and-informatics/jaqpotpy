from enum import Enum


class FeatureType(str, Enum):
    CATEGORICAL = "CATEGORICAL"
    FLOAT = "FLOAT"
    INTEGER = "INTEGER"
    SMILES = "SMILES"
    STRING = "STRING"
    TEXT = "TEXT"

    def __str__(self) -> str:
        return str(self.value)
