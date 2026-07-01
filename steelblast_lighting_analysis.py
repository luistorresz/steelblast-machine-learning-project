from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import json
import numpy as np
from sklearn.metrics import confusion_matrix

from steelblast_classic_features import (
    FeatureExtractionConfig,
    extract_features_from_image,
    normalize_illumination,
    read_grayscale_image,
)


PERTURBATIONS: tuple[str, ...] = (
    "baseline",
    "dark_25",
    "dark_50",
    "offset_plus_10",
    "offset_minus_10",
    "low_contrast",
    "high_contrast",
    "bright_25",
    "bright_50",
)

NORMAL_LIGHTING_TESTS: set[str] = {
    "dark_25",
    "dark_50",
    "offset_plus_10",
    "offset_minus_10",
    "low_contrast",
    "high_contrast",
}


@dataclass(frozen=True)
class RobustnessSummary:
    claim: str
    criterion: str
    passed: bool
    min_claim_accuracy: float
    max_claim_flip_rate: float
    note: str


def perturb_lighting(image: np.ndarray, perturbation: str) -> np.ndarray:
    if perturbation == "baseline":
        return np.clip(image, 0.0, 1.0)
    if perturbation == "dark_25":
        return np.clip(image * 0.75, 0.0, 1.0)
    if perturbation == "dark_50":
        return np.clip(image * 0.50, 0.0, 1.0)
    if perturbation == "bright_25":
        return np.clip(image * 1.25, 0.0, 1.0)
    if perturbation == "bright_50":
        return np.clip(image * 1.50, 0.0, 1.0)
    if perturbation == "offset_plus_10":
        return np.clip(image + 0.10, 0.0, 1.0)
    if perturbation == "offset_minus_10":
        return np.clip(image - 0.10, 0.0, 1.0)
    if perturbation == "low_contrast":
        return np.clip((image - 0.5) * 0.75 + 0.5, 0.0, 1.0)
    if perturbation == "high_contrast":
        return np.clip((image - 0.5) * 1.25 + 0.5, 0.0, 1.0)
    raise ValueError(f"Unknown lighting perturbation: {perturbation}")


def _safe_recall(correct: float, row_total: float) -> float:
    return float(correct / row_total) if row_total > 0 else 0.0


def evaluate_lighting_robustness(
    y_true: np.ndarray,
    predict_for_perturbation: Callable[[str], np.ndarray],
    perturbations: tuple[str, ...] = PERTURBATIONS,
    normal_lighting_tests: set[str] = NORMAL_LIGHTING_TESTS,
) -> list[dict[str, object]]:
    baseline_predictions = predict_for_perturbation("baseline")

    rows: list[dict[str, object]] = []
    for perturbation in perturbations:
        y_variant = predict_for_perturbation(perturbation)
        matrix_variant = confusion_matrix(y_true, y_variant, labels=[0, 1])
        accuracy = float(np.mean(y_variant == y_true))
        flips = int(np.sum(y_variant != baseline_predictions))
        good_recall = _safe_recall(matrix_variant[0, 0], matrix_variant[0].sum())
        not_good_recall = _safe_recall(matrix_variant[1, 1], matrix_variant[1].sum())

        rows.append(
            {
                "perturbation": perturbation,
                "accuracy": accuracy,
                "prediction_flips_vs_baseline": flips,
                "flip_rate_vs_baseline": float(flips / len(y_true)),
                "good_recall": good_recall,
                "not_good_recall": not_good_recall,
                "confusion_matrix": matrix_variant.tolist(),
                "included_in_robustness_claim": perturbation in normal_lighting_tests,
            }
        )

    return rows


def summarize_lighting_robustness(
    rows: list[dict[str, object]],
    claim: str,
    criterion: str,
    min_accuracy_threshold: float,
    max_flip_rate_threshold: float,
    note: str,
) -> dict[str, object]:
    claim_rows = [row for row in rows if bool(row["included_in_robustness_claim"])]
    min_claim_accuracy = min(float(row["accuracy"]) for row in claim_rows)
    max_claim_flip_rate = max(float(row["flip_rate_vs_baseline"]) for row in claim_rows)

    summary = RobustnessSummary(
        claim=claim,
        criterion=criterion,
        passed=bool(
            min_claim_accuracy >= min_accuracy_threshold
            and max_claim_flip_rate <= max_flip_rate_threshold
        ),
        min_claim_accuracy=float(min_claim_accuracy),
        max_claim_flip_rate=float(max_claim_flip_rate),
        note=note,
    )
    return {
        "claim": summary.claim,
        "criterion": summary.criterion,
        "passed": summary.passed,
        "min_claim_accuracy": summary.min_claim_accuracy,
        "max_claim_flip_rate": summary.max_claim_flip_rate,
        "note": summary.note,
    }


def print_lighting_table(rows: list[dict[str, object]]) -> None:
    print("perturbation       acc   flips  flip_rate  good_recall  not_good_recall")
    for row in rows:
        print(
            f"{str(row['perturbation']):16s} "
            f"{float(row['accuracy']):.3f} "
            f"{int(row['prediction_flips_vs_baseline']):5d} "
            f"{float(row['flip_rate_vs_baseline']):.3f} "
            f"{float(row['good_recall']):.3f} "
            f"{float(row['not_good_recall']):.3f}"
        )


def run_robustness_analysis(
    model_name: str,
    y_test: np.ndarray,
    predict_for_perturbation: Callable[[str], np.ndarray],
    claim: str,
    criterion: str,
    min_accuracy_threshold: float,
    max_flip_rate_threshold: float,
    note: str,
) -> dict[str, object]:
    rows = evaluate_lighting_robustness(
        y_true=y_test,
        predict_for_perturbation=predict_for_perturbation,
    )
    summary = summarize_lighting_robustness(
        rows=rows,
        claim=claim,
        criterion=criterion,
        min_accuracy_threshold=min_accuracy_threshold,
        max_flip_rate_threshold=max_flip_rate_threshold,
        note=note,
    )

    print(f"\n=== {model_name} ===")
    print_lighting_table(rows)
    print("\nRobustness summary")
    print(json.dumps(summary, indent=2))

    return {
        "summary": summary,
        "results": rows,
    }


def build_svm_predict_fn(
    image_paths: list[Path],
    feature_config: FeatureExtractionConfig,
    fitted_model,
) -> Callable[[str], np.ndarray]:
    base_images = [
        read_grayscale_image(image_path, feature_config.image_size, config=None)
        for image_path in image_paths
    ]

    def predict_for_perturbation(perturbation: str) -> np.ndarray:
        features: list[np.ndarray] = []
        for image in base_images:
            perturbed_image = perturb_lighting(image, perturbation)
            normalized_image = normalize_illumination(perturbed_image, feature_config)
            features.append(extract_features_from_image(normalized_image, feature_config))

        feature_matrix = np.vstack(features)
        return np.asarray(fitted_model.predict(feature_matrix), dtype=np.int64)

    return predict_for_perturbation


def _load_rgb_image_01(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    image = load_img(image_path, target_size=image_size)
    image = img_to_array(image).astype(np.float32) / 255.0
    return np.clip(image, 0.0, 1.0)


def build_transfer_predict_fn(
    image_paths: list[Path],
    model,
    image_size: tuple[int, int],
    preprocess_input_fn: Callable[[np.ndarray], np.ndarray],
    threshold: float = 0.5,
) -> Callable[[str], np.ndarray]:
    base_images = [_load_rgb_image_01(image_path, image_size) for image_path in image_paths]

    def predict_for_perturbation(perturbation: str) -> np.ndarray:
        batch = np.stack(
            [perturb_lighting(image, perturbation) for image in base_images],
            axis=0,
        )
        batch = preprocess_input_fn((batch * 255.0).astype(np.float32))
        probabilities = model.predict(batch, verbose=0).ravel()
        return (probabilities >= threshold).astype(np.int64)

    return predict_for_perturbation