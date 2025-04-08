import torch
import io

from jaqpot_api_client import Feature, ModelVisibility, ModelTask

from jaqpotpy.models import Model
import logging
from typing import List

logger = logging.getLogger(__name__)


class TorchONNXModel(Model):
    def __init__(
        self,
        model,
        input_example,
        task: ModelTask,
        independent_features: List[Feature],
        dependent_features: List[Feature],
        onnx_preprocessor,
    ):
        super().__init__()
        self.model = model
        self.task = task
        self.input_example = input_example
        self.onnx_bytes = None
        self.independent_features = independent_features
        self.dependent_features = dependent_features
        self.onnx_preprocessor = onnx_preprocessor

    def convert_to_onnx(self):
        logger.info("Converting model to ONNX")
        f = io.BytesIO()
        torch.onnx.export(
            self.model,
            self.input_example,
            f,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        f.seek(0)
        self.onnx_bytes = f.read()
        logger.info("Conversion complete")

    def deploy_on_jaqpot(
        self, jaqpot, name: str, description: str, visibility: ModelVisibility
    ):
        if self.onnx_bytes is None:
            self.convert_to_onnx()

        jaqpot.deploy_torch_onnx_model(
            model=self,
            name=name,
            description=description,
            visibility=visibility,
        )
