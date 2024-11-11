class JaqpotApiException(Exception):
    """Base exception class for Jaqpot API-related errors."""

    def __init__(self, message, status_code=None):
        super().__init__(f"Error {status_code}: {message}")
        self.message = message
        self.status_code = status_code

    def __str__(self):
        return f"JaqpotApiException: Status Code {self.status_code}, Message: {self.message}"


class JaqpotPredictionFailureException(JaqpotApiException):
    """Exception raised when there is a failure with the prediction."""

    def __init__(self, message):
        super().__init__("Prediction failed")
        self.message = message

    def __str__(self):
        return "JaqpotPredictionFailureException: The prediction has failed"


class JaqpotPredictionTimeoutException(Exception):
    """Exception raised when the prediction call times out."""

    def __init__(self, message):
        super().__init__("Prediction has timed out")

    def __str__(self):
        return "JaqpotPredictionTimeoutException: Prediction has timed out"
