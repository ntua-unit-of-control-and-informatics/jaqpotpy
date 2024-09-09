from enum import Enum


class DatasetCSVStatus(str, Enum):
    CREATED = "CREATED"
    EXECUTING = "EXECUTING"
    FAILURE = "FAILURE"
    SUCCESS = "SUCCESS"

    def __str__(self) -> str:
        return str(self.value)
