import io

import torch
import torch.nn.functional as F


class MyImagePreprocessor(torch.nn.Module):
    def __init__(self, target_size=(128, 128)):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        # x: [N, H, W, C] uint8
        x = x.permute(0, 3, 1, 2).float() / 255.0  # [N, C, H, W]
        x = F.interpolate(
            x, size=self.target_size, mode="bilinear", align_corners=False
        )
        return x

    def export_to_onnx(self):
        dummy_input = torch.randint(
            0, 256, (1, 256, 256, 3), dtype=torch.uint8
        )  # any size

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
