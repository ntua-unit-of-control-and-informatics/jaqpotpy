from enum import Enum


class ModelVisibility(str, Enum):
    ORG_SHARED = "ORG_SHARED"
    PRIVATE = "PRIVATE"
    PUBLIC = "PUBLIC"

    def __str__(self) -> str:
        return str(self.value)
