from enum import Enum


class ErrorCode(str, Enum):
    VALUE_0 = "1001"

    def __str__(self) -> str:
        return str(self.value)
