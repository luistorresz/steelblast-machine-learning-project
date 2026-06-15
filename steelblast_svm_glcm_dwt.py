"""Train an SVM classifier for SteelBlastQC using GLCM and DWT features.

The dataset is expected to use the official SteelBlastQC layout:

    SteelBlastQC/
      train/
        good/
        not-good/
      test/
        good/
        not-good/

Labels:
    good -> 0
    not-good -> 1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pywt
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABELS = {"good": 0, "not-good": 1}
CLASS_NAMES = ["good", "not-good"]


@dataclass(frozen=True)
class FeatureConfig:
    image_size: int = 256

    glcm_levels: int = 32
    glcm_distances: tuple[int, ...] = (1, 2, 4, 8)
    glcm_angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    glcm_properties: tuple[str, ...] = (
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    )
    dwt_wavelet: str = "db2" #?
    dwt_level: int = 3


def load_split(dataset_dir: Path, split: str) -> tuple[list[Path], np.ndarray]:
    image_paths: list[Path] = []
    labels: list[int] = []

    for class_name, label in LABELS.items():
        class_dir = dataset_dir / split / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing expected folder: {class_dir}")

        paths = sorted(class_dir.glob("*.png"))
        image_paths.extend(paths)
        labels.extend([label] * len(paths))

    if not image_paths:
        raise FileNotFoundError(f"No .png images found in {dataset_dir / split}")

    return image_paths, np.asarray(labels, dtype=np.int64)

# TODO NN won't use grayscale
def read_grayscale_image(image_path: Path, image_size: int) -> np.ndarray:
    image = imread(image_path)

    if image.ndim == 3:
        image = image[..., :3]
        image = rgb2gray(image)

    image = resize(
        image,
        (image_size, image_size),
        anti_aliasing=True,
        preserve_range=True,
    )

    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0

    return image


def quantize_image(image: np.ndarray, levels: int) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    quantized = np.floor(clipped * (levels - 1)).astype(np.uint8)
    return quantized


def extract_glcm_features(image: np.ndarray, config: FeatureConfig) -> np.ndarray:
    quantized = quantize_image(image, config.glcm_levels)
    glcm = graycomatrix(
        quantized,
        distances=config.glcm_distances,
        angles=config.glcm_angles,
        levels=config.glcm_levels,
        symmetric=True,
        normed=True,
    )

    features: list[float] = []
    for prop in config.glcm_properties:
        values = graycoprops(glcm, prop).ravel()
        features.extend(values)
        features.append(float(values.mean()))
        features.append(float(values.std()))

    return np.asarray(features, dtype=np.float32)


def describe_coefficients(coefficients: np.ndarray) -> list[float]:
    values = coefficients.ravel().astype(np.float64)
    abs_values = np.abs(values)
    energy = np.mean(values**2)
    histogram, _ = np.histogram(abs_values, bins=32, density=False)
    probabilities = histogram.astype(np.float64) / max(histogram.sum(), 1)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return [
        float(values.mean()),
        float(values.std()),
        float(abs_values.mean()),
        float(abs_values.std()),
        float(energy),
        float(entropy),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 90)),
    ]


def extract_dwt_features(image: np.ndarray, config: FeatureConfig) -> np.ndarray:
    coeffs = pywt.wavedec2(
        image,
        wavelet=config.dwt_wavelet,
        level=config.dwt_level,
        mode="symmetric",
    )

    features: list[float] = []
    approximation = coeffs[0]
    features.extend(describe_coefficients(approximation))

    for horizontal, vertical, diagonal in coeffs[1:]:
        features.extend(describe_coefficients(horizontal))
        features.extend(describe_coefficients(vertical))
        features.extend(describe_coefficients(diagonal))

    return np.asarray(features, dtype=np.float32)


def extract_features(image_path: Path, config: FeatureConfig) -> np.ndarray:
    image = read_grayscale_image(image_path, config.image_size)
    glcm_features = extract_glcm_features(image, config)
    dwt_features = extract_dwt_features(image, config)
    return np.concatenate([glcm_features, dwt_features])


def build_feature_matrix(
    image_paths: list[Path],
    config: FeatureConfig,
    split_name: str,
) -> np.ndarray:
    features = []

    for index, image_path in enumerate(image_paths, start=1):
        features.append(extract_features(image_path, config))
        if index % 100 == 0 or index == len(image_paths):
            print(f"{split_name}: extracted {index}/{len(image_paths)} images")

    return np.vstack(features)


def balanced_limit(
    image_paths: list[Path],
    labels: np.ndarray,
    limit_per_class: int,
) -> tuple[list[Path], np.ndarray]:
    selected_paths: list[Path] = []
    selected_labels: list[int] = []

    for label in sorted(np.unique(labels)):
        class_indices = np.flatnonzero(labels == label)[:limit_per_class]
        selected_paths.extend(image_paths[index] for index in class_indices)
        selected_labels.extend(labels[index] for index in class_indices)

    return selected_paths, np.asarray(selected_labels, dtype=np.int64)


def train_model(X_train: np.ndarray, y_train: np.ndarray, n_jobs: int) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced")),
        ]
    )

    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.01, 0.001, 0.0001],
    }

    min_class_count = int(np.bincount(y_train).min())
    #at most 5 folds
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError("Each class needs at least two training images.")
    #Stratified splitting keeps the same class proportions in every fold as in the whole dataset.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    #CV cross validation
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=n_jobs,
        verbose=2,
    )
    search.fit(X_train, y_train)
    return search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SteelBlastQC SVM using GLCM co-occurrence and DWT features."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("doi-10.34894-ekznn0(1)/SteelBlastQC"),
        help="Path to the SteelBlastQC dataset folder.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("steelblast_svm_glcm_dwt.joblib"),
        help="Where to save the trained model bundle.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("steelblast_svm_glcm_dwt_metrics.json"),
        help="Where to save test metrics as JSON.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resize images to image-size x image-size before feature extraction.",
    )
    parser.add_argument(
        "--quick-limit",
        type=int,
        default=None,
        help="Optional image limit per class for fast smoke tests.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for GridSearchCV. Use -1 outside restricted environments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FeatureConfig(image_size=args.image_size)

    train_paths, y_train = load_split(args.dataset_dir, "train")
    test_paths, y_test = load_split(args.dataset_dir, "test")

    if args.quick_limit is not None:
        train_paths, y_train = balanced_limit(train_paths, y_train, args.quick_limit)
        test_paths, y_test = balanced_limit(test_paths, y_test, args.quick_limit)

    print(f"Train images: {len(train_paths)}")
    print(f"Test images:  {len(test_paths)}")
    print(f"Feature config: {config}")

    X_train = build_feature_matrix(train_paths, config, "train")
    X_test = build_feature_matrix(test_paths, config, "test")

    print(f"Training feature matrix: {X_train.shape}")
    print(f"Testing feature matrix:  {X_test.shape}")

    search = train_model(X_train, y_train, args.n_jobs)
    y_pred = search.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred).tolist()

    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV F1: {search.best_score_:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix:")
    print(np.asarray(matrix))

    model_bundle = {
        "model": search.best_estimator_,
        "feature_config": asdict(config),
        "class_names": CLASS_NAMES,
        "labels": LABELS,
    }
    joblib.dump(model_bundle, args.output_model)

    metrics = {
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
        "classification_report": report,
        "confusion_matrix": matrix,
        "train_images": len(train_paths),
        "test_images": len(test_paths),
        "feature_count": int(X_train.shape[1]),
    }
    args.metrics_json.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to:   {args.output_model.resolve()}")
    print(f"Saved metrics to: {args.metrics_json.resolve()}")


if __name__ == "__main__":
    main()
