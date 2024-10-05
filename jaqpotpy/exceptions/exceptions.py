class JaqpotPredictionError(Exception):
    """Exception raised when there is an error with the prediction service."""


class JaqpotPredictionTimeout(Exception):
    """Exception raised when the prediction service times out."""


class JaqpotGetModelError(Exception):
    """Exception raised when there is an error with the get models service."""


class JaqpotApiException(Exception):
    """Exception raised when there is an error with the Jaqpot API."""
