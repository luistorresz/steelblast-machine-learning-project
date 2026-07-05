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