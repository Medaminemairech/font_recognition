import json
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms
from transform import CropTextBlockTransform

crop_transform = CropTextBlockTransform(lang_list=["en"])


def load_model(model_path: str):
    """
    loads the ONNX model and class encoding from the specified path.
    Args:
        model_path: path to the directory containing model.onnx and class_encoding.json
        Returns:
        session: ONNX Runtime InferenceSession
        class_ids: dict mapping class index -> class name
    """
    session = ort.InferenceSession(f"{model_path}/model.onnx")
    class_ids_str = json.load(open(f"{model_path}/class_encoding.json"))
    class_ids = {int(k): v for k, v in class_ids_str.items()}
    return session, class_ids


def run_inference(session, input_data):
    """
    runs inference on the input data using the provided ONNX Runtime session.
    Args:
        session: ONNX Runtime InferenceSession
        input_data: numpy array of shape (1, C, H, W), dtype float32
    Returns:
        probs: numpy array of shape (1, num_classes), dtype float32
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: input_data})[0]

    # apply softmax
    probs = np.exp(result) / np.sum(np.exp(result), axis=1, keepdims=True)

    return probs


def preprocess_input(image_path, img_size=224):
    """
    Loads an image and prepares it for inference.
    Returns a numpy array of shape (1, C, H, W), dtype float32.
    """
    transform = transforms.Compose(
        [
            crop_transform,
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor.numpy().astype(np.float32)


def postprocess_output(
    probs, class_ids: Dict[int, str]
) -> Tuple[Dict[str, float], str]:
    """
    Args:
        probs: torch.Tensor or np.ndarray of shape (num_classes,) or (1, num_classes)
        class_ids: dict mapping class index -> class name

    Returns:
        output: dict with keys "class_proba" (dict of class probabilities) and "predicted_class" (str)
    """
    # Convert to 1D numpy array
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    probs = probs.squeeze()  # remove batch dim if present
    output = {}
    output["class_proba"] = {class_ids[i]: float(probs[i]) for i in range(len(probs))}
    predicted_class = class_ids[int(np.argmax(probs))]
    output["predicted_class"] = predicted_class

    return output
