# Day 4 Summary Report

## Overview
Day 4 focused on **differential privacy (DP)** training, class imbalance handling, and **bias evaluation**.

We trained head-only models with:
- **Baseline DP** (ε ≈ 8, ~97.5% accuracy).
- **Stronger DP** (ε ≈ 0.2, ~97% accuracy).
- **Weighted DP** (ε ≈ 0.2, with class-weighted loss to address minority class imbalance).

We then conducted evaluation runs, including a **bias-aware analysis** for the "supportive" class.

---

## Key Metrics

### Comparison Table

| Model Variant        | ε (epsilon) | Accuracy | Precision (w) | Recall (w) | F1 (w) |
|----------------------|-------------|----------|---------------|------------|--------|
| DP (baseline)        | ~8.0        | 0.976    | 0.976         | 0.976      | 0.975  |
| DP (strong)          | ~0.19       | 0.971    | 0.971         | 0.971      | 0.969  |
| DP (strong+weighted) | ~0.19       | 0.971    | 0.971         | 0.971      | 0.970  |

---

### Bias Analysis (supportive class underprediction)

- Supportive recall: 0.556  
- Misgendering recall: 0.800  
- Harassment recall: 0.993  
- Outing/doxxing recall: 0.963  

Macro-average recall dropped (0.828), confirming recall bias in the smallest class.

---

## Findings
1. **Privacy–utility tradeoff**:  
   - Stronger privacy (ε ~0.2) reduced accuracy only slightly compared to ε ~8.
   - DP head-only models remain performant even under strong privacy.

2. **Class imbalance**:  
   - Weighted loss improved fairness, but supportive recall remains low.
   - This suggests model bias against the supportive class persists, likely due to tiny sample size.

3. **Bias-aware eval**:  
   - Highlighted supportive class as a weak spot despite strong overall performance.
   - Confusion matrix showed many supportive examples misclassified as harassment.

---

## Next Steps (Day 5 Preview)
- Explore **full model DP training** (not just head-only).
- Investigate **robustness under distribution shift**.
- Consider **data augmentation** or **re-sampling** to boost supportive class recall.
- Continue documenting privacy–bias tradeoffs.

---

**Artifacts Produced**
- day4_dp_head_metrics.json
- day4_dp_head_metrics_strong.json
- day4_dp_head_metrics_strong_weighted.json
- day4_dp_head_eval_report.txt
- day4_dp_head_confusion_matrix.json
- day4_final_eval_report.txt
- day4_final_confusion_matrix.json

