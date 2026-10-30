# ORCA Model Evaluation & Benchmarking Report

- **Champion Model Selected**: Gradient Boosting
- **Training Dataset**: 17,049 historical observations (Years 2000–2016)
- **Validation Dataset**: 2,273 observations (Years 2015–2016)
- **Evaluation Dataset**: 5,678 temporal holdout observations (Years 2017–2021) from California Current Ecosystem (CalCOFI / OBIS)
- **Test Accuracy**: 0.9544
- **Test Macro Precision**: 0.9594
- **Test Macro Recall**: 0.9501
- **Test Macro F1-Score**: 0.9542
- **Multi-Class ROC-AUC (OvR)**: 0.9965

### Per-Class Detailed Performance

| Ecosystem Condition | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Healthy** | 0.940 | 0.994 | 0.966 | 2348 |
| **Moderate Stress** | 0.946 | 0.917 | 0.931 | 1913 |
| **High Stress** | 0.992 | 0.939 | 0.965 | 1417 |

### Candidate Model Benchmark

| Model | 5-Fold CV Macro F1 | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9598 | 0.9567 | 0.9594 | 0.9533 | 0.9561 | 0.9951 |
| Decision Tree | 0.9580 | 0.9449 | 0.9511 | 0.9414 | 0.9456 | 0.9849 |
| Random Forest | 0.9668 | 0.9526 | 0.9576 | 0.9481 | 0.9522 | 0.9966 |
| Gradient Boosting | 0.9690 | 0.9544 | 0.9594 | 0.9501 | 0.9542 | 0.9965 |

### Scientific Explainability & Contributing Factors

Top environmental predictors ranked by tree importance:
- **phosphate**: 47.15%
- **sigma_theta_proxy**: 34.68%
- **nitrate**: 5.90%
- **salinity**: 4.74%
- **temp_depth_gradient**: 2.65%
- **temperature**: 0.94%
- **longitude**: 0.93%
- **silicate**: 0.90%

### Limitations & Methodological Integrity
- **Observational Scope**: Sensor profiles reflect the California Current System; extrapolating to Arctic or tropical coral reefs requires regional calibration.
- **Correlation vs. Causation**: ML features reflect statistical associations with ecological distress, not laboratory-proven causality.
- **Sampling Depth**: Data density is highest in the upper 200m euphotic zone where photosynthetic activity and pelagic fisheries occur.
