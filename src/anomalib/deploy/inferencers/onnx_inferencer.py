"""ONNXRuntime Inferencer for optimized model inference.

This module provides the ONNX inferencer implementation for running optimized
inference with ONNX models using onnxruntime.

Example:
    Assume we have an ONNX model file in the following structure:

        $ tree weights
        ./weights
        ├── model.onnx
        └── metadata.json

    Create an ONNX inferencer:

    >>> from anomalib.deploy import ONNXInferencer
    >>> inferencer = ONNXInferencer(
    ...     path="weights/model.onnx",
    ...     device="GPU"
    ... )

    Make predictions:

    >>> # From image path
    >>> prediction = inferencer.predict("path/to/image.jpg")

    >>> # From PIL Image
    >>> from PIL import Image
    >>> image = Image.open("path/to/image.jpg")
    >>> prediction = inferencer.predict(image)

    >>> # From numpy array (Single Image: HWC)
    >>> import numpy as np
    >>> image = np.random.rand(224, 224, 3)
    >>> prediction = inferencer.predict(image)

    >>> # From batched numpy array (Variable Batch Size: NHWC)
    >>> images = np.random.rand(4, 224, 224, 3)
    >>> prediction = inferencer.predict(images)

    The prediction result contains anomaly maps and scores:

    >>> prediction.anomaly_map  # doctest: +SKIP
    array([[[0.1, 0.2, ...]]], dtype=float32)

    >>> prediction.pred_score  # doctest: +SKIP
    array([0.86, 0.92, ...])
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning_utilities.core.imports import module_available
from PIL.Image import Image as PILImage

from anomalib.data import NumpyImageBatch
from anomalib.data.utils import read_image

logger = logging.getLogger("anomalib")


class ONNXInferencer:
    """ONNX inferencer for optimized model inference using ONNXRuntime.

    Args:
        path (str | Path | bytes): Path to the ONNX model file (``.onnx``) or
            model data as bytes.
        device (str | None, optional): Inference device.
            Options: ``"CPU"``, ``"GPU"``, ``"CUDA"``.
            Defaults to ``"GPU"``.
        providers (list[str] | None, optional): Specific ONNXRuntime execution
            providers. If provided, overrides the `device` argument.
            Defaults to ``None``.

    Example:
        >>> from anomalib.deploy import ONNXInferencer
        >>> model = ONNXInferencer(
        ...     path="model.onnx",
        ...     device="GPU"
        ... )
        >>> prediction = model.predict("test.jpg")
    """

    def __init__(
        self,
        path: str | Path | bytes,
        device: str | None = "GPU",
        providers: list[str] | None = None,
    ) -> None:
        if not module_available("onnxruntime"):
            msg = "onnxruntime is not installed. Please install onnxruntime-gpu to use ONNXInferencer."
            raise ImportError(msg)

        self.device = device
        
        # Configure execution providers
        if providers is not None:
            self.providers = providers
        else:
            if self.device and self.device.upper() in {"GPU", "CUDA"}:
                self.providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self.providers = ["CPUExecutionProvider"]

        self.inputs, self.outputs, self.session = self.load_model(path)

    def load_model(self, path: str | Path | bytes) -> tuple[Any, Any, Any]:
        """Load ONNX model from file or bytes.

        Args:
            path (str | Path | bytes): Path to ONNX model file or model
                data as bytes.

        Returns:
            tuple[Any, Any, Any]: Tuple containing:
                - Model inputs metadata
                - Model outputs metadata
                - ONNXRuntime InferenceSession

        Raises:
            ValueError: If model path has invalid extension.
        """
        import onnxruntime as ort

        # If bytes are passed
        if isinstance(path, bytes):
            session = ort.InferenceSession(path, providers=self.providers)
        else:
            path = path if isinstance(path, Path) else Path(path)
            if path.suffix != ".onnx":
                msg = f"Path must be an .onnx file. Got {path.suffix}"
                raise ValueError(msg)
            session = ort.InferenceSession(str(path), providers=self.providers)

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        return inputs, outputs, session

    @staticmethod
    def pre_process(image: np.ndarray) -> np.ndarray:
        """Pre-process input image or batch of images.

        Args:
            image (np.ndarray): Input image or batch of images.

        Returns:
            np.ndarray: Pre-processed image(s) with shape (N,C,H,W).
        """
        # Normalize numpy array to range [0, 1]
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0

        # Add batch dimension if it is a single image (H, W, C)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # Transpose from (N, H, W, C) to (N, C, H, W)
        if image.shape[-1] in {1, 3}:
            image = image.transpose(0, 3, 1, 2)

        return image

    def post_process(self, predictions: list[np.ndarray]) -> dict:
        """Convert ONNXRuntime predictions to dictionary.

        Args:
            predictions (list[np.ndarray]): Raw output arrays from the ONNX session.

        Returns:
            dict: Dictionary of prediction tensors mapped to their output names.
        """
        names = [output.name for output in self.outputs]
        return dict(zip(names, predictions, strict=False))

    def predict(self, image: str | Path | np.ndarray | PILImage | torch.Tensor) -> NumpyImageBatch:
        """Run inference on an input image or batch of images.

        Args:
            image (str | Path | np.ndarray | PILImage | torch.Tensor): Input image(s) 
                as file path or array. If passing a numpy array or torch tensor, 
                it can have variable batch size (N, H, W, C) or (N, C, H, W).

        Returns:
            NumpyImageBatch: Batch containing the predictions.

        Raises:
            TypeError: If image input is invalid type.
        """
        # Convert file path or string to image if necessary
        if isinstance(image, str | Path):
            image = read_image(image, as_tensor=False)
        elif isinstance(image, PILImage):
            image = np.array(image) / 255.0
        elif isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        image = self.pre_process(image)
        
        # Prepare feed_dict mapping input node names to the pre-processed image array.
        # This assumes the primary input node is the first one.
        feed_dict = {self.inputs[0].name: image}
        
        # Run inference
        # The first argument is None to request all outputs
        predictions = self.session.run(None, feed_dict)
        
        # Map back to a dict using the output names
        pred_dict = self.post_process(predictions)

        return NumpyImageBatch(image=image, **pred_dict)
