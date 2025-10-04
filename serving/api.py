import os
import logging
from fastapi import FastAPI, Query, HTTPException
from utils import load_model, preprocess_input, run_inference, postprocess_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model once on startup
model, class_ids = load_model(model_path="model/smallresnet")
logger.info("Model loaded successfully.")


@app.get("/predict")
def predict(image_path: str = Query(..., description="Path to the input image")):
    """
    Predict endpoint that takes an image path and returns class probabilities.
    Example:
        GET /predict?image_path=path/to/image.jpg
    """
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail=f"Image not found: {image_path}")

    try:
        # Load and preprocess image
        logger.info(f"Processing image: {image_path}")
        input_tensor = preprocess_input(image_path)  # shape: [1, C, H, W]

        # Run inference
        logger.info("Running inference...")
        probs = run_inference(model, input_tensor)  # shape: [1, num_classes]

        # Convert to JSON
        logger.info("Postprocessing output...")
        output = postprocess_output(probs, class_ids)

        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Run the server directly
    uvicorn.run(
        "api:app",  # still use "module:app" even in Python
        host="0.0.0.0",
        port=8000,
        reload=True,  # auto-reload on code changes
        log_level="info",
    )
