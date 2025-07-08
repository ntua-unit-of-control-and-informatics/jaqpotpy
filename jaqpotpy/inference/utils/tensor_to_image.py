"""
Simple tensor to image conversion without torchvision dependency.

This module provides tensor to base64 image conversion that avoids
torchvision dependencies which can cause operator registration issues.
"""

import base64
import io
import numpy as np
import torch
from PIL import Image


def tensor_to_base64_img_simple(tensor: torch.Tensor) -> str:
    """
    Convert a 3D tensor (C, H, W) to a base64 PNG image without torchvision.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W) with 1 or 3 channels

    Returns:
        str: Base64 encoded PNG image with data URI prefix

    Raises:
        ValueError: If tensor shape is invalid
    """
    if tensor.ndim != 3 or tensor.size(0) not in [1, 3]:
        raise ValueError("Expected tensor of shape (C, H, W) with 1 or 3 channels")

    # Convert to numpy and ensure proper format
    tensor = tensor.detach().cpu().clamp(0, 1)
    np_array = tensor.numpy()

    # Convert from (C, H, W) to (H, W, C)
    np_array = np.transpose(np_array, (1, 2, 0))

    # Convert to 0-255 range
    np_array = (np_array * 255).astype(np.uint8)

    # Handle single channel (grayscale)
    if np_array.shape[2] == 1:
        np_array = np_array.squeeze(2)
        mode = "L"
    else:
        mode = "RGB"

    # Create PIL image
    image = Image.fromarray(np_array, mode=mode)

    # Save to buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode to base64
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"
