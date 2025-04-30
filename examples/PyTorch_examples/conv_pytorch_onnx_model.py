import torch
from jaqpot_api_client import ModelTask, Feature, FeatureType, ModelVisibility
from examples.PyTorch_examples.preprocessors.my_image_preprocessor import (
    MyImagePreprocessor,
)
from jaqpotpy.models.torch_models.torch_onnx import TorchONNXModel
from jaqpotpy import Jaqpot


class LargeModel(torch.nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()

        layers = []

        in_channels = 3
        out_channels = 9
        num_layers = 13

        for _ in range(num_layers):
            layers.append(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            layers.append(torch.nn.ReLU())
            in_channels = out_channels  # so next layer matches input

        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(out_channels * 128 * 128, 1024))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(1024, 3))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Generate large input tensor (still reasonable for testing)
input_tensor = torch.rand((1, 3, 128, 128), dtype=torch.float32)

# Create model and ONNX preprocessor
model = LargeModel()
preprocessor = MyImagePreprocessor()
onnx_preprocessor = preprocessor.export_to_onnx()

independent_features = [
    Feature(key="input", name="Input", feature_type=FeatureType.IMAGE)
]
dependent_features = [
    Feature(key="image", name="Image", feature_type=FeatureType.IMAGE)
]

# Upload to Jaqpot
jaqpot = Jaqpot()
jaqpot.login()

# Build ONNX wrapper model
jaqpot_model = TorchONNXModel(
    model,
    input_tensor,
    ModelTask.REGRESSION,
    independent_features=independent_features,
    dependent_features=dependent_features,
    onnx_preprocessor=onnx_preprocessor,
)

# This will trigger the large model upload logic due to ONNX size
jaqpot_model.deploy_on_jaqpot(
    jaqpot,
    name="Large Torch ONNX Model",
    description="A large test model to trigger S3 upload flow",
    visibility=ModelVisibility.PRIVATE,
)
