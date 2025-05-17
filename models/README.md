# Model Directory - miRNA Target Site Classification

This directory contains the best-performing models trained for miRNA target site classification using various deep learning architectures, including ResNet variants with and without regularisation.

## 📊 Model Performance Summary (Averaged Across Datasets)

| Model Name                                  | Architecture            | Reg. | Avg F1 | PR-AUC | Accuracy |
| ------------------------------------------- | ----------------------- | ---- | ------ | ------ | -------- |
| **Manakov2022\_dr0.13.keras**               | ResNet Medium (No Reg)  | ❌    | 0.715  | 0.801  | 72.6%    |
| **Manakov2022\_dr0.09.keras**               | ResNet Small (No Reg)   | ❌    | 0.714  | 0.776  | 72.7%    |
| **Manakov2022\_dr0.13.keras**               | ResNet Large (No Reg)   | ❌    | 0.710  | 0.784  | 72.0%    |
| **L1L2\_Hejret2023\_dr0.25\_rf0.002.keras** | ResNet Large (With Reg) | ✅    | 0.692  | 0.779  | 70.6%    |
| **Hejret2023\_dr0.09.keras**                | ResNet Large (No Reg)   | ❌    | 0.681  | 0.725  | 66.9%    |

---

## 📁 Folder Structure

```plaintext
models/
├── best_model/
│   └── Manakov2022_dr0.13.keras              # Top overall
├── small_model/
│   └── Manakov2022_dr0.09.keras              # Lightweight & performant
├── large_model/
│   └── Manakov2022_dr0.13.keras              # High-capacity variant
├── regularized_model/
│   └── L1L2_Hejret2023_dr0.25_rf0.002.keras  # For generalization
└── consistent_model/
    └── Hejret2023_dr0.09.keras               # Stable performance
```

---

## 📝 Notes

* All models are saved in **Keras format** (`.keras`).
* Regularised models include L1 and L2 constraints and were tuned to balance overfitting.
* Metrics reflect average performance over multiple real-world test datasets.

Feel free to experiment, compare results, or fine-tune further depending on your application!
