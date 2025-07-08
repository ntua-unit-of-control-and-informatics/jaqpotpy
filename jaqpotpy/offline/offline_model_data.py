from typing import Optional, Any, Dict


class OfflineModelData:
    """
    Container for downloaded model data from Jaqpot platform.

    This class provides a clean interface for accessing downloaded model components
    including ONNX model bytes, preprocessor, and metadata.
    """

    def __init__(
        self,
        model_id: int,
        model_metadata: Any,
        onnx_bytes: bytes,
        preprocessor: Optional[Any] = None,
    ):
        """
        Initialize OfflineModelData.

        Args:
            model_id: The model ID from Jaqpot platform
            model_metadata: Model metadata object from Jaqpot API
            onnx_bytes: Raw ONNX model bytes
            preprocessor: Deserialized preprocessor object (optional)
        """
        self.model_id = model_id
        self.model_metadata = model_metadata
        self.onnx_bytes = onnx_bytes
        self.preprocessor = preprocessor

    @property
    def has_preprocessor(self) -> bool:
        """
        Check if model has a preprocessor.

        Returns:
            True if preprocessor is available, False otherwise
        """
        return self.preprocessor is not None

    @property
    def independent_features(self) -> list:
        """
        Get independent features from model metadata.

        Returns:
            List of independent features, empty list if not available
        """
        return getattr(self.model_metadata, "independent_features", [])

    @property
    def dependent_features(self) -> list:
        """
        Get dependent features from model metadata.

        Returns:
            List of dependent features, empty list if not available
        """
        return getattr(self.model_metadata, "dependent_features", [])

    @property
    def task(self) -> Optional[Any]:
        """
        Get model task from metadata.

        Returns:
            Model task, or None if not available
        """
        return getattr(self.model_metadata, "task", None)

    @property
    def model_type(self) -> Optional[str]:
        """
        Get model type from metadata.

        Returns:
            Model type, or None if not available
        """
        return getattr(self.model_metadata, "type", None)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format (for backward compatibility).

        Returns:
            Dictionary with model data components
        """
        return {
            "model_id": self.model_id,
            "model_metadata": self.model_metadata,
            "onnx_bytes": self.onnx_bytes,
            "preprocessor": self.preprocessor,
        }

    def __repr__(self) -> str:
        """
        String representation of OfflineModelData.

        Returns:
            String representation
        """
        return (
            f"OfflineModelData("
            f"model_id={self.model_id}, "
            f"model_type={self.model_type}, "
            f"has_preprocessor={self.has_preprocessor}, "
            f"onnx_size={len(self.onnx_bytes)} bytes)"
        )
