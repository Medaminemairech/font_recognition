from training.config import EvalConfig
from training.evaluation import Evaluator


def evaluate_model(predicitions_csv_path: str):
    evaluator = Evaluator(EvalConfig(predictions_csv_path=predicitions_csv_path))
    out = evaluator.evaluate()
    return out


if __name__ == "__main__":
    predicitions_csv_path = "c:/Users/amine/Downloads/PUBLICIS/training/artifacts/predictions/smallresnet/best_predictions.csv"
    evaluate_model(predicitions_csv_path)
