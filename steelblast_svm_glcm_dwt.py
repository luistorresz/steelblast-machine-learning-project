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
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    image_size: int = 256 # Resize images to image-size x image-size before feature extraction. This standardization is important for consistent feature extraction, especially for GLCM and DWT, which can be sensitive to image dimensions. A size of 256x256 provides a good balance between retaining detail and keeping computational requirements manageable.

    glcm_levels: int = 32 # The number of gray levels to quantize the image into for GLCM calculation. Reducing to 32 levels helps manage the size of the co-occurrence matrix and focuses on broader texture patterns, which can be beneficial for classification while keeping computational complexity reasonable. TODO validate that with paper
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

# load either train or test images 
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
        image = rgb2gray(image) # Convert to grayscale using luminosity method, which accounts for human perception of color brightness. This is important for texture analysis with GLCM, as it relies on intensity values. The resulting grayscale image will have values in the range [0, 1], which is suitable for further processing and feature extraction.

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


def read_display_image(image_path: Path, image_size: int) -> np.ndarray:
    image = imread(image_path)

    if image.ndim == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=2)
    else:
        image = image[..., :3]

    image = resize(
        image,
        (image_size, image_size),
        anti_aliasing=True,
        preserve_range=True,
    )

    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0

    return np.clip(image, 0.0, 1.0)


def quantize_image(image: np.ndarray, levels: int) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    quantized = np.floor(clipped * (levels - 1)).astype(np.uint8)
    return quantized


def extract_glcm_features(image: np.ndarray, config: FeatureConfig) -> np.ndarray:
    #reduce to 32 intensity levels
    quantized = quantize_image(image, config.glcm_levels)
    # Co-occurrence Matrix
    glcm = graycomatrix(
        quantized,
        distances=config.glcm_distances,
        angles=config.glcm_angles,
        levels=config.glcm_levels,
        symmetric=True,
        normed=True,
    )

    features: list[float] = []
    # extract 6 configured features
    for prop in config.glcm_properties:
        values = graycoprops(glcm, prop).ravel()
        features.extend(values)
        features.append(float(values.mean()))
        features.append(float(values.std()))

    return np.asarray(features, dtype=np.float32)

# DWT coefficients can have a wide range of values, including negative and positive numbers. To capture the texture information effectively, we compute several statistics on the coefficients, such as mean, standard deviation, energy (mean of squared values), and entropy (which measures the randomness or complexity of the coefficients). Additionally, we include percentiles to capture the distribution of coefficient values. These features help summarize the texture information contained in the DWT coefficients in a way that is useful for classification.
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
    # Apply wavelet decomposition
    coeffs = pywt.wavedec2(
        image,
        wavelet=config.dwt_wavelet,
        level=config.dwt_level,
        mode="symmetric",
    )

    features: list[float] = []
    approximation = coeffs[0]
    features.extend(describe_coefficients(approximation))

    # Repeat across 3 levels
    for horizontal, vertical, diagonal in coeffs[1:]:
        features.extend(describe_coefficients(horizontal))
        features.extend(describe_coefficients(vertical))
        features.extend(describe_coefficients(diagonal))

    return np.asarray(features, dtype=np.float32)


def extract_features(image_path: Path, config: FeatureConfig) -> np.ndarray:
    image = read_grayscale_image(image_path, config.image_size)
    return extract_features_from_image(image, config)

# get Final feature vector
def extract_features_from_image(image: np.ndarray, config: FeatureConfig) -> np.ndarray:
    glcm_features = extract_glcm_features(image, config)
    dwt_features = extract_dwt_features(image, config)
    return np.concatenate([glcm_features, dwt_features])

# Loops through all images
# Extracts features
# Stacks them into matrix:
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

# Computes confidence score for each prediction
def predicted_label_confidences(model: Pipeline, features: np.ndarray) -> np.ndarray:
    predictions = model.predict(features)
    # decision_function gives distance to the separating hyperplane. For binary classification, it's a single score where positive means class 1 and negative means class 0. Magnitude of distance = confidence
    scores = model.decision_function(features)
    # 
    if scores.ndim == 1:
        positive_class = int(model.classes_[1])
        signed_scores = np.where(predictions == positive_class, scores, -scores)
        return signed_scores.astype(np.float64)

    return np.asarray(
        [scores[index, np.flatnonzero(model.classes_ == label)[0]] for index, label in enumerate(predictions)],
        dtype=np.float64,
    )


def prediction_confidence(model: Pipeline, features: np.ndarray, label: int) -> float:
    scores = model.decision_function(features.reshape(1, -1))

    if np.ndim(scores) == 1:
        positive_class = int(model.classes_[1])
        score = float(scores[0])
        return score if label == positive_class else -score

    class_index = int(np.flatnonzero(model.classes_ == label)[0])
    return float(scores[0, class_index])

# Mask parts of the image and see how prediction confidence changes.
# Steps:
# 1. Compute base prediction confidence
# 2. Slide a patch over the image
# 3. Replace patch with median value
# 4. Recompute prediction
# 5. Measure confidence drop
def compute_occlusion_focus_heatmap(
    image: np.ndarray,
    model: Pipeline,
    predicted_label: int,
    config: FeatureConfig,
    patch_size: int, # The size of the square region (in pixels) that gets “hidden” at each step.
    stride: int, # The number of pixels the patch moves horizontally and vertically between steps. Smaller strides create smoother heatmaps but require more computations.
) -> np.ndarray:
    base_features = extract_features_from_image(image, config)
    base_confidence = prediction_confidence(model, base_features, predicted_label)
    heatmap = np.zeros_like(image, dtype=np.float32)
    counts = np.zeros_like(image, dtype=np.float32)
    fill_value = float(np.median(image))
    height, width = image.shape

    for top in range(0, height, stride):
        bottom = min(top + patch_size, height)
        top = max(0, bottom - patch_size)

        for left in range(0, width, stride):
            right = min(left + patch_size, width)
            left = max(0, right - patch_size)

            occluded = image.copy()
            occluded[top:bottom, left:right] = fill_value
            occluded_features = extract_features_from_image(occluded, config)
            occluded_confidence = prediction_confidence(model, occluded_features, predicted_label)
            importance = max(0.0, base_confidence - occluded_confidence)

            heatmap[top:bottom, left:right] += importance
            counts[top:bottom, left:right] += 1

    return np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts > 0)


def save_focus_pair(
    display_image: np.ndarray,
    heatmap: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    axes[0].imshow(display_image)
    axes[0].set_title("Example image")
    axes[0].set_axis_off()

    axes[1].imshow(display_image)
    vmax = float(heatmap.max()) if heatmap.size and heatmap.max() > 0 else 1.0
    overlay = axes[1].imshow(heatmap, cmap="jet", alpha=0.62, vmin=0, vmax=vmax)
    axes[1].set_title("Activation heatmap")
    axes[1].set_axis_off()
    fig.colorbar(overlay, ax=axes[1], label="Decision-score drop")
    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_classification_case_heatmaps(
    test_paths: list[Path],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    X_test: np.ndarray,
    model: Pipeline,
    config: FeatureConfig,
    output_dir: Path,
    patch_size: int,
    stride: int,
) -> dict[str, dict[str, object]]:
    heatmap_paths: dict[str, dict[str, object]] = {}
    confidences = predicted_label_confidences(model, X_test)
    case_definitions = {
        "true_positive": (1, 1),
        "false_positive": (0, 1),
        "false_negative": (1, 0),
        "true_negative": (0, 0),
    }

    # iterate over TP/FP/FN/TN cases, find the most confident example of each, generate heatmap, and save results
    for case_name, (actual_label, predicted_label) in case_definitions.items():
        # find indices of all test cases that match the actual and predicted labels for this case type
        case_indices = [
            index
            for index, (actual, predicted) in enumerate(zip(y_test, y_pred))
            if actual == actual_label and predicted == predicted_label
        ]
        # sort those indices by confidence score, highest first, so the most confident example is at the front
        case_indices = sorted(case_indices, key=lambda index: confidences[index], reverse=True)
        case_dir = output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        # select the most confident example for this case type, if any exist
        selected_index = case_indices[0] if case_indices else None
        output_path = case_dir / f"{case_name}_focus_heatmap.png"
        source_image = None
        confidence = None
        max_activation = None

        if selected_index is not None:
            image_path = test_paths[selected_index]
            grayscale_image = read_grayscale_image(image_path, config.image_size)
            display_image = read_display_image(image_path, config.image_size)
            heatmap = compute_occlusion_focus_heatmap(
                grayscale_image,
                model,
                int(y_pred[selected_index]),
                config,
                patch_size,
                stride,
            )
            title = (
                f"{case_name.replace('_', ' ').title()} | "
                f"actual {CLASS_NAMES[int(y_test[selected_index])]}, "
                f"predicted {CLASS_NAMES[int(y_pred[selected_index])]}"
            )
            # save the side-by-side image and heatmap visualization
            save_focus_pair(display_image, heatmap, title, output_path)
            source_image = str(image_path)
            confidence = float(confidences[selected_index])
            max_activation = float(heatmap.max())
        # record the results for this case type, including whether a heatmap was generated and the confidence/activation values
        heatmap_paths[case_name] = {
            "actual_label": CLASS_NAMES[actual_label],
            "predicted_label": CLASS_NAMES[predicted_label],
            "available_images": len(case_indices),
            "generated_images": int(selected_index is not None),
            "source_image": source_image,
            "heatmap": str(output_path) if selected_index is not None else None,
            "confidence": confidence,
            "max_activation": max_activation,
        }
        print(heatmap_paths[case_name])

    return heatmap_paths


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

# train model using SVM (RBF kernel)
def train_model(X_train: np.ndarray, y_train: np.ndarray, n_jobs: int) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()), # StandardScaler standardizes features by removing the mean and scaling to unit variance, which is important for SVM performance since it relies on distances in feature space. This ensures that all features contribute equally to the model and prevents features with larger scales from dominating the decision boundary.
            ("svm", SVC(kernel="rbf", class_weight="balanced")), # The SVC with RBF kernel is a powerful non-linear classifier that can capture complex relationships in the data. The class_weight="balanced" option automatically adjusts weights inversely proportional to class frequencies, which helps address any class imbalance in the dataset and can improve performance on the minority class.
        ]
    )
    # Define the hyperparameter grid for GridSearchCV
    param_grid = {
        "svm__C": [0.1, 1, 10, 100], # The C parameter controls the trade-off between achieving a low training error and a low testing error (generalization). A smaller C encourages a simpler decision boundary that may misclassify some training points but generalizes better, while a larger C tries to classify all training points correctly, which can lead to overfitting. Testing a range of values allows us to find the optimal balance for our dataset.
        "svm__gamma": ["scale", 0.01, 0.001, 0.0001], # The gamma parameter defines how far the influence of a single training example reaches. A low gamma means that the model considers points at a larger distance from the decision boundary, resulting in a smoother decision surface. A high gamma focuses more on points close to the decision boundary, which can lead to a more complex model that may overfit. Including "scale" allows the model to automatically adjust gamma based on the number of features, which can be a good default choice.
    }

    min_class_count = int(np.bincount(y_train).min())
    #at most 5 folds
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError("Each class needs at least two training images.")
    #Stratified splitting keeps the same class proportions in every fold as in the whole dataset.
    #random_state ensures reproducibility, and shuffle=True randomizes the data before splitting to avoid any order bias.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    #CV cross validation
    #GridSearchCV exhaustively tries all combinations of the specified hyperparameters (C and gamma for the SVM) and evaluates each combination using cross-validation on the training data. It selects the combination that yields the best average F1 score across the folds.
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
        "--heatmap-dir",
        type=Path,
        default=Path("steelblast_svm_glcm_dwt_focus_heatmaps"),
        help="Directory where TP/FP/FN/TN focus heatmap PNG files will be saved.",
    )
    parser.add_argument(
        "--occlusion-patch-size",
        type=int,
        default=32,
        help="Square patch size for occlusion-sensitivity focus heatmaps.",
    )
    parser.add_argument(
        "--occlusion-stride",
        type=int,
        default=16,
        help="Stride between occlusion patches. Smaller values give smoother but slower heatmaps.",
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
    # for heatmap generation, we need a reasonable patch size and stride to see meaningful differences in confidence when occluding parts of the image. Enforce positive integers for these parameters.
    if args.occlusion_patch_size <= 0:
        raise ValueError("--occlusion-patch-size must be greater than zero.")
    if args.occlusion_stride <= 0:
        raise ValueError("--occlusion-stride must be greater than zero.")

    
 

    config = FeatureConfig(image_size=args.image_size)

    train_paths, y_train = load_split(args.dataset_dir, "train")
    test_paths, y_test = load_split(args.dataset_dir, "test")

    # for a quick training
    if args.quick_limit is not None:
        train_paths, y_train = balanced_limit(train_paths, y_train, args.quick_limit)
        test_paths, y_test = balanced_limit(test_paths, y_test, args.quick_limit)

    print(f"Train images: {len(train_paths)}")
    print(f"Test images:  {len(test_paths)}")
    print(f"Feature config: {config}")

    X_train = build_feature_matrix(train_paths, config, "train")
    X_test = build_feature_matrix(test_paths, config, "test")

    MODEL_PATH = "steelblast_svm_glcm_dwt.joblib"

    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = joblib.load(MODEL_PATH)
    else:

        print(f"Training feature matrix: {X_train.shape}")
        print(f"Testing feature matrix:  {X_test.shape}")

        model = train_model(X_train, y_train, args.n_jobs)

    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred).tolist()

    print(f"Best parameters: {model.best_params_}")
    print(f"Best CV F1: {model.best_score_:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix:")
    print(np.asarray(matrix))

    heatmap_paths = save_classification_case_heatmaps(
        test_paths,
        y_test,
        y_pred,
        X_test,
        model.best_estimator_,
        config,
        args.heatmap_dir,
        args.occlusion_patch_size,
        args.occlusion_stride,
    )


    metrics = {
        "best_params": model.best_params_,
        "best_cv_f1": float(model.best_score_),
        "classification_report": report,
        "confusion_matrix": matrix,
        "train_images": len(train_paths),
        "test_images": len(test_paths),
        "feature_count": int(X_train.shape[1]),
        "heatmap_method": "occlusion_sensitivity",
        "heatmap_settings": {
            "heatmaps_per_case": 1,
            "occlusion_patch_size": args.occlusion_patch_size,
            "occlusion_stride": args.occlusion_stride,
        },
        "heatmaps": heatmap_paths,
    }
    args.metrics_json.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to:   {args.output_model.resolve()}")
    print(f"Saved metrics to: {args.metrics_json.resolve()}")
    print(f"Saved heatmaps to: {args.heatmap_dir.resolve()}")


if __name__ == "__main__":
    main()
