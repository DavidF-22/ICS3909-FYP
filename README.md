
# miRBind_2.0: miRNA Target Site Prediction

## Project Overview
miRBind is a deep learning-based project focused on predicting microRNA (miRNA) target site binding. The project explores different neural network architectures to enhance the accuracy of miRNA-mRNA interaction predictions. This work is part of ongoing research in bioinformatics, specifically in the area of post-transcriptional gene regulation.

## Models Implemented
The repository contains the following models for miRNA target site classification:
- **Residual Network (ResNet)**
- **Bidirectional Long Short-Term Memory (BiLSTM)**
- **Deep Recurrent Neural Network (DeepRNN)**

## Cloning the Repository
To clone this repository, use the following command:
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

## Setting Up the Environment
> _**Note:** Some of the libraries used in this project are only compatible with Mac, Linux or Windows Subsystem for Linux (WSL). Running the project on native Windows may cause compatibility issues._

1. Ensure you are using a Linux-based system or WSL.
2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Shell Scripts
Before running any shell script in the repository, you must grant execution permissions. For example, to set permissions for the `encode.sh` script in the ResNet model, run:
```bash
chmod +x ResNet/code/encode.sh
```
To execute the script, use:
```bash
ResNet/code/encode.sh
```

## Script Input Requirements
- **`encode.sh`**  
  This script requires **no inputs**.

- **Training and Evaluation Scripts**  
  These scripts require **two inputs**:
  1. The first input should be either `NoReg` or `WithReg`, indicating whether to train without or with regularization.
  2. The second input should be either `plot_true` or `plot_false`, specifying if plots should be generated.

- **Prediction Scripts**  
  These scripts require only **one input**: either `NoReg` or `WithReg`.

>_**Note:** Please ensure that you supply the correct inputs corresponding to the desired operation, as mismatches may affect the naming conventions and subsequent mapping of models to plots and results._

## Model Naming Conventions
When training the models, the programs generate multiple models based on the chosen hyperparameters. There are two scenarios:

1.  **Using both regularization factors and dropout rates:**
    ```python
    reg_factors = [0.01, 0.005, 0.005, 0.01, 0.003, 0.002] 
    dropout_rates = [0.05, 0.09, 0.13, 0.17, 0.21, 0.25]
    ```
This configuration produces 6 models for each regularizer type (L1, L2, and L1L2), for a total of **18 models**.

2. **Using only dropout rates:**
    ```python
    dropout_rates = [0.05, 0.09, 0.13, 0.17, 0.21, 0.25]
    ```
This configuration produces **6 models**.

Each model is named with a number at the end (1 through 6) reflecting the specific combination of regularization factor and dropout rate. In addition, a "plot number" is assigned alongside the model number. These numbers are used in the text file output by the programs to map the trained models to their corresponding plots, and subsequently, to the results.

## Datasets and Data Folder Placeholders
Within the repository, several data folders include placeholder `README.md` files. These placeholders indicate, by file name, which datasets need to be placed in each folder. 

**Note:** The files `AGO2_eCLIP_Manakov2022_leftout.tsv` and `AGO2_eCLIP_Manakov2022_full_dataset.tsv` should not be used.

### Dataset Sources
All required datasets can be obtained from the following link: [miRBench_Datasets - Zendo](https://zenodo.org/records/14501607)

### Unpacking .tsv.gz Files
This repository works with `.tsv` files. If your datasets are provided as `.tsv.gz` files, they will need to be unpacked:

- **Windows:**
You can extract `.gz` files using the `tar` command in Command Prompt or by installing the [7-Zip](https://www.7-zip.org/) program.

- **Mac**
Simply double-click the file to extract it, or use the command in a Terminal window:
  ```bash
  gunzip filename.gz
  ```

- **Linux**
Use the following command to decompress the file:
  ```bash
  gzip -d `__filename__`.gz
  ```

---
This project is part of ongoing research in miRNA target site prediction and aims to improve the understanding of miRNA-mRNA interactions using deep learning techniques.
