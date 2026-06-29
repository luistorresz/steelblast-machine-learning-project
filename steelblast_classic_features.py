from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pywt
from skimage import exposure
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
from skimage.transform import resize


LABELS = {"good": 0, "not-good": 1}
CLASS_NAMES = ["good", "not-good"]


@dataclass(frozen=True)
class FeatureExtractionConfig:
    image_size: int = 256
    illumination_normalization: str = "clahe"
    clahe_clip_limit: float = 0.01
    contrast_percentiles: tuple[float, float] = (2.0, 98.0)
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
    dwt_wavelet: str = "db2"
    dwt_level: int = 3


def extract_features_from_image(image: np.ndarray, config: object) -> np.ndarray:
    """Build the final feature vector by concatenating GLCM and DWT features."""
    glcm_features = extract_glcm_features(image, config)
    dwt_features = extract_dwt_features(image, config)
    return np.concatenate([glcm_features, dwt_features])


def extract_glcm_features(image: np.ndarray, config: object) -> np.ndarray:
    quantized = quantize_image(image, int(config.glcm_levels))
    glcm = graycomatrix(
        quantized,
        distances=tuple(config.glcm_distances),
        angles=tuple(config.glcm_angles),
        levels=int(config.glcm_levels),
        symmetric=True,
        normed=True,
    )

    features: list[float] = []
    for prop in tuple(config.glcm_properties):
        values = graycoprops(glcm, prop).ravel()
        features.extend(values)
        features.append(float(values.mean()))
        features.append(float(values.std()))

    return np.asarray(features, dtype=np.float32)


def extract_dwt_features(image: np.ndarray, config: object) -> np.ndarray:
    coeffs = pywt.wavedec2(
        image,
        wavelet=str(config.dwt_wavelet),
        level=int(config.dwt_level),
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


def quantize_image(image: np.ndarray, levels: int) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    quantized = np.floor(clipped * (levels - 1)).astype(np.uint8)
    return quantized


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


def normalize_illumination(image: np.ndarray, config: object) -> np.ndarray:
    image = np.clip(image.astype(np.float32), 0.0, 1.0)
    method = str(config.illumination_normalization)

    if method == "none":
        return image

    if method == "clahe":
        return exposure.equalize_adapthist(
            image,
            clip_limit=float(config.clahe_clip_limit),
        ).astype(np.float32)

    if method == "percentile":
        lower, upper = np.percentile(image, tuple(config.contrast_percentiles))
        if upper <= lower + 1e-6:
            return image
        return np.clip((image - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)

    raise ValueError(
        "illumination_normalization must be one of: none, clahe, percentile."
    )


def read_grayscale_image(
    image_path: Path,
    image_size: int,
    config: object | None = None,
) -> np.ndarray:
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

    if config is None:
        return image

    return normalize_illumination(image, config)
