"""
Image processing utilities for PyTorch models.

This module contains utilities for handling image data in ONNX inference,
including validation, conversion, and tensor operations.
"""

import base64
import io
from typing import Union
import torch
from PIL import Image, UnidentifiedImageError
from PIL.ImageFile import ImageFile
import torchvision.transforms as T


def validate_and_decode_image(b64_string: str) -> ImageFile:
    """
    Validate and decode a base64 encoded image.

    Args:
        b64_string (str): Base64 encoded image string

    Returns:
        ImageFile: PIL Image object in RGB format

    Raises:
        ValueError: If the image is invalid or cannot be decoded
    """
    try:
        image_bytes = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.verify()  # Validate format, throws if not valid image

        # Reopen image to use (verify() exhausts the file)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError("Invalid image input") from e


def tensor_to_base64_img(tensor: torch.Tensor) -> str:
    """
    Convert a 3D tensor (C, H, W) to a base64 PNG image after validation.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W) with 1 or 3 channels

    Returns:
        str: Base64 encoded PNG image with data URI prefix

    Raises:
        ValueError: If tensor shape is invalid or generated image is corrupt
    """
    if tensor.ndim != 3 or tensor.size(0) not in [1, 3]:
        raise ValueError("Expected tensor of shape (C, H, W) with 1 or 3 channels")

    # Clamp values to [0, 1] and convert to PIL image
    tensor = tensor.detach().cpu().clamp(0, 1)
    to_pil = T.ToPILImage()
    image = to_pil(tensor)

    # Save to in-memory buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Validate the generated image
    try:
        validated = Image.open(buffer).convert("RGB")
        validated.verify()  # Raises error if corrupt
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Generated image is not valid") from e

    # Return base64 encoded image with data URI prefix
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def preprocess_image_for_model(
    image: Union[ImageFile, str],
    target_size: tuple = (224, 224),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Preprocess an image for PyTorch model inference.

    Args:
        image: PIL Image object or base64 string
        target_size: Target size for resizing (width, height)
        normalize: Whether to apply ImageNet normalization

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    if isinstance(image, str):
        image = validate_and_decode_image(image)

    # Define transforms
    transforms = [
        T.Resize(target_size),
        T.ToTensor(),
    ]

    if normalize:
        transforms.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    transform = T.Compose(transforms)
    return transform(image).unsqueeze(0)  # Add batch dimension
