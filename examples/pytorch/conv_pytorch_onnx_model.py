import torch
from jaqpot_api_client import ModelTask, Feature, FeatureType, ModelVisibility

from examples.pytorch.preprocessors.my_image_preprocessor import (
    MyImagePreprocessor,
)
from jaqpotpy.models.torch_models.torch_onnx import TorchONNXModel


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 5)  # <- only 3 output channels

    def forward(self, x):
        return torch.sigmoid(self.conv1(x))  # normalize to [0, 1] for image


input_tensor = torch.rand((1, 3, 128, 128), dtype=torch.float32)

model = MyModel()
preprocessor = MyImagePreprocessor()
onnx_preprocessor = preprocessor.export_to_onnx()

independent_features = list(
    [Feature(key="input", name="Input", feature_type=FeatureType.IMAGE)]
)
dependent_features = list(
    [Feature(key="image", name="Image", feature_type=FeatureType.IMAGE)]
)

# Upload model on Jaqpot
# First import Jaqpot class from jaqpotpy
from jaqpotpy import Jaqpot  # noqa: E402

# Next, create an instance of Jaqpot
jaqpot = Jaqpot()
jaqpot_model = TorchONNXModel(
    model,
    input_tensor,
    ModelTask.REGRESSION,
    independent_features=independent_features,
    dependent_features=dependent_features,
    onnx_preprocessor=onnx_preprocessor,
)
jaqpot.login()
jaqpot_model.deploy_on_jaqpot(
    jaqpot,
    name="Torch ONNX Model v12",
    description="Torch description",
    visibility=ModelVisibility.PUBLIC,
)
