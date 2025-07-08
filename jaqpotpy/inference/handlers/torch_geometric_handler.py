"""
Torch Geometric prediction handler - isolated from main prediction logic.

This handler deals with the specific requirements of torch geometric models
including graph featurization and model inference.
"""

import numpy as np
import torch
import io
from typing import List, Dict, Any
import onnxruntime
from jaqpot_api_client import ModelType
from jaqpot_api_client.models.dataset import Dataset


def handle_torch_geometric_prediction(model_data, dataset: Dataset) -> np.ndarray:
    """
    Handle torch geometric prediction with custom featurization logic.

    This bypasses the standard dataset creation to avoid compatibility issues
    with the hardcoded graph featurizer requirements.

    Args:
        model_data: OfflineModelData containing model and metadata
        dataset: Raw dataset with SMILES input

    Returns:
        np.ndarray: Predictions from the torch geometric model
    """
    # Load featurizer from torch_config
    feat_config = model_data.model_metadata.torch_config
    featurizer = _load_featurizer(feat_config)

    # Extract SMILES data directly from dataset input
    user_input = []
    for row in dataset.input:
        # Remove jaqpotRowId and keep the actual data
        row_data = {k: v for k, v in row.items() if k != "jaqpotRowId"}
        user_input.append(row_data)

    predictions = []

    if model_data.model_type == ModelType.TORCH_GEOMETRIC_ONNX:
        # Use raw ONNX bytes
        onnx_model = model_data.onnx_bytes

        for inp in user_input:
            # Featurize the SMILES string into graph data
            graph_data = featurizer.featurize(inp["SMILES"])
            model_output = _torch_geometric_onnx_predict(onnx_model, graph_data)
            predictions.append(model_output)

    elif model_data.model_type == ModelType.TORCHSCRIPT:
        # Use raw TorchScript bytes
        torchscript_model = model_data.onnx_bytes

        for inp in user_input:
            # Featurize the SMILES string into graph data
            graph_data = featurizer.featurize(inp["SMILES"])
            model_output = _torchscript_predict(torchscript_model, graph_data)
            predictions.append(model_output)
    else:
        raise ValueError(
            f"Unsupported torch geometric model type: {model_data.model_type}"
        )

    # Convert to numpy array
    if predictions:
        return torch.stack(predictions).detach().numpy()
    else:
        return np.array([])


def _torch_geometric_onnx_predict(onnx_model_bytes: bytes, graph_data) -> torch.Tensor:
    """Run ONNX inference for torch geometric models."""
    ort_session = onnxruntime.InferenceSession(onnx_model_bytes)

    # Prepare inputs for ONNX model
    ort_inputs = {
        ort_session.get_inputs()[0].name: _to_numpy(graph_data.x),
        ort_session.get_inputs()[1].name: _to_numpy(graph_data.edge_index),
        ort_session.get_inputs()[2].name: _to_numpy(
            torch.zeros(graph_data.x.shape[0], dtype=torch.int64)
        ),
    }

    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)
    return torch.tensor(ort_outputs[0])


def _torchscript_predict(torchscript_bytes: bytes, graph_data) -> torch.Tensor:
    """Run TorchScript inference for torch geometric models."""
    # Load TorchScript model from bytes
    model_buffer = io.BytesIO(torchscript_bytes)
    model_buffer.seek(0)
    torchscript_model = torch.jit.load(model_buffer)
    torchscript_model.eval()

    # Run inference
    with torch.no_grad():
        if hasattr(graph_data, "edge_attr") and graph_data.edge_attr.shape[1] == 0:
            output = torchscript_model(
                graph_data.x, graph_data.edge_index, graph_data.batch
            )
        else:
            output = torchscript_model(
                graph_data.x,
                graph_data.edge_index,
                graph_data.batch,
                graph_data.edge_attr,
            )

    return output


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def _load_featurizer(config: Dict[str, Any]):
    """Load the graph featurizer from configuration."""
    from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer

    featurizer = SmilesGraphFeaturizer()
    featurizer.load_dict(config)
    featurizer.sort_allowable_sets()
    return featurizer
