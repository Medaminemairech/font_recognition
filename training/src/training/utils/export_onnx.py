import os
from pathlib import Path
import json
import torch
from training.config import ModelConfig
from training.models import models_dict


def export_to_onnx(
    model_name,
    checkpoint_path: str,
    export_path: str,
    input_size: tuple = (1, 3, 224, 224),
):

    state_dict = torch.load(checkpoint_path)
    idx_to_class = {v: k for k, v in state_dict["classes_dict"].items()}
    # Create model
    model = models_dict[model_name](
        ModelConfig(name=model_name, num_classes=state_dict["number_of_classes"])
    )
    # Load checkpoint

    model.model.load_state_dict(state_dict["model_state_dict"])
    model.model.eval()

    export_path = os.path.join(export_path, f"{model_name}/")
    export_path = Path(export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(input_size)

    # Export to ONNX
    torch.onnx.export(
        model.model,
        dummy_input,
        f"{export_path}/model.onnx",  # where to save the model
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],  # input layer name
        output_names=["output"],  # output layer name
        dynamic_axes={
            "input": {0: "batch"},  # allow variable batch size
            "output": {0: "batch"},
        },
    )

    # Save class encoding
    # Save alongside ONNX
    with open(f"{export_path}/class_encoding.json", "w") as f:
        json.dump(idx_to_class, f)
    print(f"Model exported to {export_path}/{model_name}.onnx")
