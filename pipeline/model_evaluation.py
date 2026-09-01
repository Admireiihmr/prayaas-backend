import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from prayaas.config.configuration import CLASS_LABELS, settings
from prayaas.logger import logger
from prayaas.utils.common import save_json

CANCER, NORMAL, OPMD = 0, 1, 2
BINARY_LABELS = [CANCER, NORMAL]
BINARY_NAMES = ["CANCER", "NON CANCER"]


def _scores(matrix: np.ndarray, i: int) -> dict:
    """Per-class precision/recall/f1 read off the confusion matrix.

    Derived here rather than via precision_recall_fscore_support, whose return
    type varies with `average` and so cannot be indexed cleanly. Verified against
    classification_report, which is still logged below.
    """
    true_positive = int(matrix[i, i])
    predicted = int(matrix[:, i].sum())
    actual = int(matrix[i, :].sum())

    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": actual,
    }


def collapse_to_binary(predicted_class: int) -> int:
    """Maps the model's 3 classes onto the 2 folders that exist on disk.

    OPMD is premalignant, not confirmed cancer, so where it belongs is a clinical
    judgement rather than an obvious default. OPMD_IS_CANCER in .env decides it.
    """
    if predicted_class == OPMD:
        return CANCER if settings.opmd_is_cancer else NORMAL
    return predicted_class


class ModelEvaluation:
    """Scores the labelled images in raw_data against the pretrained model."""

    def run(self, index: pd.DataFrame, predictions: list[dict]) -> dict:
        df = index.copy()
        df["predicted_raw"] = [p["predicted_class"] for p in predictions]
        df["confidence"] = [p["confidence"] for p in predictions]
        df["predicted"] = df["predicted_raw"].map(collapse_to_binary)

        y_true, y_pred = df["label"], df["predicted"]
        correct = int((y_true == y_pred).sum())

        matrix = confusion_matrix(y_true, y_pred, labels=BINARY_LABELS)

        metrics = {
            "n_images": int(len(df)),
            "accuracy": round(correct / len(df), 4),
            "opmd_is_cancer": settings.opmd_is_cancer,
            "per_class": {
                name: _scores(matrix, i) for i, name in enumerate(BINARY_NAMES)
            },
            # How often the model reached for the class that has no ground truth.
            "predicted_opmd_count": int((df["predicted_raw"] == OPMD).sum()),
            "raw_prediction_distribution": {
                CLASS_LABELS[i]: int((df["predicted_raw"] == i).sum()) for i in range(3)
            },
            "confusion_matrix": matrix.tolist(),
        }

        save_json(settings.metrics_path, metrics)
        df.to_csv(settings.artifacts_dir / "predictions.csv", index=False)

        logger.info("Accuracy %.4f on %s images", metrics["accuracy"], metrics["n_images"])
        logger.info(
            "\n%s",
            classification_report(y_true, y_pred, labels=BINARY_LABELS, target_names=BINARY_NAMES),
        )
        return metrics
