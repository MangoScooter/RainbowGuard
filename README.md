RainbowGuard
# 🌈 RainbowGuard

RainbowGuard is a transformer-based classification model designed to detect and evaluate online text for supportive vs. harmful content.
It is built with the Hugging Face `transformers` library and fine-tuned using differential privacy techniques to balance **accuracy**, **fairness**, and **privacy guarantees**.

---

## 🚀 Features
- **Transformer backbone (HF Trainer)**
- **Differential Privacy (DP) training** with multiple ε levels
- **Bias-aware evaluation** (per-class analysis, focus on minority classes)
- **Threshold tuning** for optimized F1
- **Reproducible environment** (`requirements.txt` pinned)

---

## 📊 Key Metrics (Day 4 Summary)

| Model Variant   | ε (epsilon) | Accuracy | Precision (w) | Recall (w) | F1 (w) |
|-----------------|-------------|----------|---------------|------------|--------|
| DP (baseline)   | ~8.0        | 0.976    | 0.976         | 0.976      | 0.975  |
| DP (strong)     | ~0.2        | 0.970    | 0.970         | 0.970      | 0.969  |
| DP (weighted)   | ~0.2        | 0.972    | 0.972         | 0.972      | 0.971  |

---

## 📂 Repository Structure
