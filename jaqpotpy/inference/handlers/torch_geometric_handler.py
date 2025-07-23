"""
Torch Geometric prediction handler - isolated from main prediction logic.

This handler deals with the specific requirements of torch geometric models
including graph featurization and model inference.
"""

import numpy as np
import torch
import io

import onnxruntime
from jaqpot_api_client import ModelType, ModelTask
from jaqpot_api_client.models.dataset import Dataset

from jaqpotpy.descriptors.graph import SmilesGraphFeaturizer
from jaqpotpy.offline.offline_model_data import OfflineModelData


def handle_torch_geometric_prediction(model_data: OfflineModelData, dataset: Dataset):
    feat_config = model_data.model_metadata.torch_config
    featurizer = _load_featurizer(feat_config)
    target_name = model_data.model_metadata.dependent_features[0].name
    model_task = model_data.model_metadata.task
    user_input = dataset.input
    raw_model = model_data.onnx_bytes

    predictions = []

    if model_data.model_metadata.type == ModelType.TORCH_GEOMETRIC_ONNX:
        for inp in user_input:
            model_output = torch_geometric_onnx_post_handler(
                raw_model, featurizer.featurize(inp["SMILES"])
            )
            predictions.append(
                generate_prediction_response(model_task, target_name, model_output, inp)
            )
    elif model_data.model_metadata.type == ModelType.TORCHSCRIPT:
        for inp in user_input:
            model_output = torchscript_post_handler(
                raw_model, featurizer.featurize(inp["SMILES"])
            )
            predictions.append(
                generate_prediction_response(model_task, target_name, model_output, inp)
            )
    return predictions


def torch_geometric_onnx_post_handler(onnx_model, data):
    ort_session = onnxruntime.InferenceSession(onnx_model)
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(data.x),
        ort_session.get_inputs()[1].name: to_numpy(data.edge_index),
        ort_session.get_inputs()[2].name: to_numpy(
            torch.zeros(data.x.shape[0], dtype=torch.int64)
        ),
    }
    ort_outs = torch.tensor(np.array(ort_session.run(None, ort_inputs)))
    return ort_outs


def torchscript_post_handler(torchscript_model, data):
    model_buffer = io.BytesIO(torchscript_model)
    model_buffer.seek(0)
    torchscript_model = torch.jit.load(model_buffer)
    torchscript_model.eval()
    with torch.no_grad():
        if data.edge_attr.shape[1] == 0:
            out = torchscript_model(data.x, data.edge_index, data.batch)
        else:
            out = torchscript_model(data.x, data.edge_index, data.batch, data.edge_attr)
    return out


def _load_featurizer(config):
    featurizer = SmilesGraphFeaturizer()
    featurizer.load_dict(config)
    featurizer.sort_allowable_sets()
    return featurizer


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def torch_binary_classification(target_name, output, inp):
    proba = torch.nn.functional.sigmoid(output).squeeze().tolist()
    pred = int(proba > 0.5)
    # UI Results
    results = {
        "jaqpotMetadata": {
            "probabilities": [round((1 - proba), 3), round(proba, 3)],
            "jaqpotRowId": inp["jaqpotRowId"],
        }
    }
    if "jaqpotRowLabel" in inp:
        results["jaqpotMetadata"]["jaqpotRowLabel"] = inp["jaqpotRowLabel"]
    results[target_name] = pred
    return results


def torch_regression(target_name, output, inp):
    pred = [output.squeeze().tolist()]
    results = {"jaqpotMetadata": {"jaqpotRowId": inp["jaqpotRowId"]}}
    if "jaqpotRowLabel" in inp:
        results["jaqpotMetadata"]["jaqpotRowLabel"] = inp["jaqpotRowLabel"]
    results[target_name] = pred
    return results


def generate_prediction_response(model_task, target_name, out, row_id):
    if model_task == ModelTask.BINARY_CLASSIFICATION:
        return torch_binary_classification(target_name, out, row_id)
    elif model_task == ModelTask.REGRESSION:
        return torch_regression(target_name, out, row_id)
    else:
        raise ValueError(
            "Only BINARY_CLASSIFICATION and REGRESSION tasks are supported"
        )
