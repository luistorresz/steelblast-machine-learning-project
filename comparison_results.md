# Models comparison
## 1. Performance Comparison

| Metric | ResNet50 | SVM (GLCM+DWT) | Better |
|----------|----------:|----------:|----------|
| Accuracy | 0.9320 | **0.9360** | SVM |
| Weighted F1 | 0.9313 | **0.9361** | SVM |
| Macro F1 | 0.9301 | **0.9355** | SVM |
| Not-good Recall | 0.8571 | **0.9464** | SVM |
| Good Recall | **0.9928** | 0.9275 | ResNet50 |

### Confusion Matrices

**ResNet50**

| Actual / Predicted | Good | Not-good |
|-------------------|-----:|----------:|
| Good | 137 | 1 |
| Not-good | 16 | 96 |

**SVM (GLCM+DWT)**

| Actual / Predicted | Good | Not-good |
|-------------------|-----:|----------:|
| Good | 128 | 10 |
| Not-good | 6 | 106 |

### Interpretation

- SVM achieves slightly better overall performance across all aggregate metrics.
- ResNet50 produces fewer false alarms (1 FP vs. 10 FP).
- SVM detects substantially more defective (*not-good*) samples (106 vs. 96), reducing missed defects from 16 to 6.
- Since missed defects are typically more costly in quality-control applications, SVM provides the more favorable error profile.

**Winner: SVM**

---

## 2. Computational Cost

| Metric | ResNet50 | SVM | Better |
|----------|----------:|----------:|----------|
| Preprocessing | **0.244 s** | 106.594 s | ResNet50 |
| Training + Validation | 554.146 s | **137.717 s** | SVM |
| Total Before Evaluation | 554.390 s | **244.311 s** | SVM |
| Inference (250 samples) | 8.133 s | **5.398 s** *| SVM |

* SVM inference consists of 0.0173 s classification time plus 5.3809 s feature extraction time.

### Interpretation

- ResNet50 requires substantially longer training.
- SVM incurs a high feature-extraction cost, but classification itself is nearly instantaneous.
- Overall, SVM is faster both for model development and inference.

**Winner: SVM**

---

## 3. Code Complexity

| Stage | SVM | ResNet50 |
|----------|----------:|----------:|
| Preprocessing | 6 | 10 |
| Modelling & Validation | 5 | 7 |
| Evaluation | 2 | 2 |
| **Total** | **13** | **19** |

### Interpretation

- The transfer-learning pipeline is more complex (19 vs. 13 cyclomatic complexity).
- Additional complexity comes from dataset handling, augmentation, and training-control logic.
- The SVM implementation is simpler and easier to maintain, test, and deploy.

**Winner: SVM**

---

## 4. Bias Analysis

### Bias Under Lighting Perturbations

#### Key Findings

- Across moderate lighting perturbations (darkening, brightening, intensity offsets, and contrast variations), the **SVM (GLCM + DWT)** model demonstrates greater stability, maintaining higher accuracy and generally exhibiting lower or comparable prediction flip rates.
- **Stress test:** Under severe overexposure (**bright_50**), **ResNet50** is substantially more robust. Although both models experience performance degradation, ResNet50 shows a smaller decline in accuracy and a considerably lower flip rate than SVM.

#### Implications for Quality Control

- **SVM (GLCM + DWT)** is the preferred choice in well-controlled steel-blast inspection environments, as it is largely insensitive to realistic lighting variations.
- **ResNet50** is more fault-tolerant when illumination conditions cannot be reliably controlled, particularly under severe overexposure.

---

### Bias Between Classes

#### ResNet50

- ResNet50 exhibits a clear bias toward the **Good** class.
- **Good recall:** 0.9928
- **Not-good recall:** 0.8571
- **Recall gap:** 0.1357, indicating a substantial imbalance in class-wise performance.
- **Interpretation:** ResNet50 is conservative when identifying defects. Consequently, it minimizes false alarms on good parts but is more likely to miss actual defective samples.

#### SVM (GLCM + DWT)

- SVM is considerably more balanced, with a slight bias toward the **Not-good** class.
- **Good recall:** 0.9275
- **Not-good recall:** 0.9464
- **Recall gap:** 0.0189, which is significantly smaller than that of ResNet50.
- **Interpretation:** SVM is more willing to flag potential defects, leading to higher defect detection rates at the cost of an increased number of false positives.


#### Practical Implications for Quality Control

- When **missing a defect is the primary risk**, **SVM (GLCM + DWT)** provides the safer bias profile because of its stronger ability to detect defective parts.
- When **false rejections are more costly than missed defects**, **ResNet50** provides the safer bias profile because it is less likely to classify good parts as defective.

---



## 4. Overall Assessment

1.	For most quality control applications, SVM is better overall.
2.	For highest not-good detection and stronger aggregate metrics in this latest run: SVM is ahead.
3.	For stricter false-alarm control on good parts: ResNet50 still has an advantage.
4.	For constrained hardware and strict latency, SVM is better, while for not constrained hardware and non critical latency, ResNet50 could be preferred.
5.	For maintainability and operational simplicity, SVM is strong.


### Final Verdict

**SVM (GLCM+DWT)** is the stronger overall solution in the current evaluation, offering:
- Higher accuracy and F1 scores,
- Better defect detection,
- Lower computational cost,
- Lower implementation complexity.