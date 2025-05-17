# Model Directory - miRNA Target Site Classification

This directory contains the "optimal" models trained for miRNA target site classification using multiple deep learning architectures, including ResNet variants with and without regularisation.

## 📊 Model Performance Summary (Averaged Across Datasets)

| Model Name                                  | Architecture            | Reg. | Avg F1 | PR-AUC | Accuracy |
| ------------------------------------------- | ----------------------- | ---- | ------ | ------ | -------- |
| **ResNet_medium_NoReg_AGO2_-eCLIP_Manakov2022_dr0.13**               | ResNet Medium (No Reg)  | ❌    | 0.715  | 0.801  | 72.6%    |
| **ResNet_small_NoReg_AGO2_-eCLIP_Manakov2022_dr0.09**               | ResNet Small (No Reg)   | ❌    | 0.714  | 0.776  | 72.7%    |
| **ResNet_large_NoReg_AGO2_-eCLIP_Manakov2022_dr0.13**               | ResNet Large (No Reg)   | ❌    | 0.710  | 0.784  | 72.0%    |
| **ResNet_large_L1L2_AGO2_-CLASH_Hejret2023_dr0.25_rf0.002** | ResNet Large (With Reg) | ✅    | 0.692  | 0.779  | 70.6%    |
| **ResNet_large_NoReg_AGO2_-CLASH_Hejret2023_dr0.09**                | ResNet Large (No Reg)   | ❌    | 0.681  | 0.725  | 66.9%    |

---

## 📁 Folder Structure

```bash
.
├── ResNet_medium_NoReg_AGO2_eCLIP_Manakov2022_dr0.13.keras        # Top Overall            
├── ResNet_small_NoReg_AGO2_eCLIP_Manakov2022_dr0.09.keras         # Lightweight & Performant
├── ResNet_large_NoReg_AGO2_eCLIP_Manakov2022_dr0.13.keras         # High-capacity variant
├── ResNet_large_L1L2_AGO2_CLASH_Hejret2023_dr0.25_rf0.002.keras   # For generalization
└── ResNet_large_NoReg_AGO2_CLASH_Hejret2023_dr0.09.keras          # Stable performance
```

---

## 📝 Notes

- All models are saved in **Keras format** (`.keras`).
- Metrics reflect average performance over multiple real-world test datasets [miRBench_Datasets - Zenodo](https://zenodo.org/records/14501607).

Feel free to experiment, compare results, or fine-tune further depending on your application!
