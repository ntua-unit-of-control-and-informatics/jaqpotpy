import os
import shutil
import random

"""Prepare dataset"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 6)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

"""Upload to Jaqpot

1. Export trained model to ONNX
"""

# Load best model if not already loaded
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
# Dummy input to trace model shape: [1, 3, 224, 224]
dummy_input = torch.randn(1, 3, 224, 224).to(
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

"""2. Create and export image preprocessor to ONNX"""

import io

import torch.nn as nn
from PIL import Image


class MyImagePreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def forward(self, x):
        # Assume x is already a [1, 3, 224, 224] tensor (handled by platform)
        return x

    def export_to_onnx(self):
        dummy_input = torch.randn(1, 3, 224, 224)

        f = io.BytesIO()
        torch.onnx.export(
            self,
            dummy_input,
            f,
            input_names=["image"],
            output_names=["tensor"],
            opset_version=11,
            dynamic_axes={"image": {1: "height", 2: "width", 0: "batch_size"}},
        )

        f.seek(0)
        return f.read()


"""3. Upload model to Jaqpot"""

from jaqpotpy.models.torch_models import TorchONNXModel
from jaqpotpy import Jaqpot
from jaqpot_api_client import ModelTask, Feature, FeatureType, ModelVisibility

# Create preprocessor object and export
preprocessor = MyImagePreprocessor()
onnx_preprocessor = preprocessor.export_to_onnx()

# Define features
independent_features = [
    Feature(key="input", name="Input Image", feature_type=FeatureType.IMAGE)
]

dependent_features = [
    Feature(key="class", name="Predicted Class", feature_type=FeatureType.STRING)
]

# Load exported model as torch.Tensor (or just use path)
input_tensor = torch.randn(1, 3, 224, 224)

# Create Jaqpot instance
jaqpot = Jaqpot()
# jaqpot.login()

model.cpu()

# Define model for upload
jaqpot_model = TorchONNXModel(
    model,
    input_tensor,
    task=ModelTask.MULTICLASS_CLASSIFICATION,
    independent_features=independent_features,
    dependent_features=dependent_features,
    onnx_preprocessor=onnx_preprocessor,
)

jaqpot_model.convert_to_onnx()
print(len(jaqpot_model.onnx_bytes))

# Upload to Jaqpot
jaqpot_model.deploy_on_jaqpot(
    jaqpot=jaqpot,
    name="Nanomaterial Classifier v1",
    description="Classifies nanomaterial types from TEM images.",
    visibility=ModelVisibility.PUBLIC,
)
