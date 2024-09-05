from enum import Enum


class OrganizationVisibility(str, Enum):
    PRIVATE = "PRIVATE"
    PUBLIC = "PUBLIC"

    def __str__(self) -> str:
        return str(self.value)
