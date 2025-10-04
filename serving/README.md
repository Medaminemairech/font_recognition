# Serving

This folder contains the code and configuration for serving trained models via a FastAPI web API.

## Structure

- [`api.py`](api.py): FastAPI application exposing a `/predict` endpoint for inference.
- [`utils.py`](utils.py): Utility functions for model loading, preprocessing, inference, and postprocessing.
- [`transform.py`](transform.py): Custom image transforms (e.g., cropping text blocks).
- [`requirements.txt`](requirements.txt): Python dependencies for serving.
- [`model/`](model/): Folder containing exported ONNX models and class encodings.
  - `smallresnet/model.onnx`: ONNX model file.
  - `smallresnet/class_encoding.json`: Mapping from class indices to class names.

## Usage

1. **Install dependencies**  
   ```sh
   uv venv .venv && .venv\Scripts\Activate && pip install -r requirements.txt
   ```

2. **Start the API server**  
   ```sh
   python api.py
   ```

3. **Make predictions**  
   Send a GET request to `/predict` with the image path:
   ```
   GET /predict?image_path=path/to/image.jpg
   ```

   The response will include class probabilities and the predicted class.

## Model Export

Export trained models to ONNX format using [`training/src/training/utils/export_onnx.py`](training/src/training/utils/export_onnx.py). Place the exported model and class encoding in the [`model/`](serving/model/) folder.

## Notes

- The API expects image paths accessible to the server.
- Update `model_path` in [`api.py`](serving/api.py) if you use a different model.