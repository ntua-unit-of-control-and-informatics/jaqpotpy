import torch
from jaqpot_api_client import ModelTask, Feature, FeatureType, ModelVisibility

from examples.PyTorch_examples.preprocessors.my_image_preprocessor import (
    MyImagePreprocessor,
)
from jaqpotpy.models.torch_models.torch_onnx import TorchONNXModel


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 5)

    def forward(self, x):
        return torch.relu(self.conv1(x))


input_tensor = torch.rand((1, 3, 128, 128), dtype=torch.float32)

model = MyModel()
preprocessor = MyImagePreprocessor()
onnx_preprocessor = preprocessor.export_to_onnx()

independent_features = list(
    [Feature(key="image", name="Image", feature_type=FeatureType.IMAGE)]
)


# Upload model on Jaqpot
# First import Jaqpot class from jaqpotpy
from jaqpotpy import Jaqpot  # noqa: E402

# Next, create an instance of Jaqpot
jaqpot = Jaqpot(
    base_url="http://localhost.jaqpot.org",
    app_url="http://localhost.jaqpot.org:3000",
    login_url="http://localhost.jaqpot.org:8070",
    api_url="http://localhost.jaqpot.org:8080",
    keycloak_realm="jaqpot-local",
    keycloak_client_id="jaqpot-local-test",
)
jaqpot_model = TorchONNXModel(
    model,
    input_tensor,
    ModelTask.REGRESSION,
    independent_features=independent_features,
    dependent_features=independent_features,
    onnx_preprocessor=onnx_preprocessor,
)
jaqpot.login()
jaqpot_model.deploy_on_jaqpot(
    jaqpot,
    name="Torch ONNX Model v4",
    description="Torch description",
    visibility=ModelVisibility.PUBLIC,
)
