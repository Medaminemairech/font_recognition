import argparse
import logging

from training.config import TestConfig
from training.inference import TestRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_inference(model_name: str, checkpoint_path: str):
    """Runs inference using the specified model and checkpoint."""
    logger.info(
        f"Running inference with model: {model_name}, checkpoint: {checkpoint_path}"
    )
    config = TestConfig(model_name=model_name, checkpoint_path=checkpoint_path)
    runner = TestRunner(config)
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to use."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    args = parser.parse_args()
    run_inference(model_name=args.model_name, checkpoint_path=args.checkpoint_path)
