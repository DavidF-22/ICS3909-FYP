
# Development of Machine Learning Methods for microRNA target site classification

## Table of Contents

- [Project Overview](#project-overview)  
- [Models Implemented](#models-implemented)  
- [Project Tree](#project-tree)  
- [Cloning the Repository](#cloning-the-repository)  
- [Setting Up the Environment](#setting-up-the-environment)  
- [Running the Shell Scripts](#running-the-shell-scripts)  
- [Script Input Requirements](#script-input-requirements)  
  - [ResNet_workflow.sh](#resnet_workflowsh)  
  - [DeepRNN_workflow.sh and BiLSTM_workflow.sh](#deeprnn_workflowsh-and-bilstm_workflowsh)  
- [Datasets and Data Folder Placeholders](#datasets-and-data-folder-placeholders)  
  - [Dataset Sources](#dataset-sources)  
  - [Unpacking .tsv.gz Files](#unpacking-tsvgz-files)

## Project Overview
This final year project (FYP) is a deep learning-based project focused on predicting microRNA (miRNA) target site binding. The project explores different neural network architectures to enhance the accuracy of miRNA-mRNA interaction predictions. This work is part of ongoing research in bioinformatics, specifically in the area of post-transcriptional gene regulation.

This project builds on the [miRBind project](https://www.mdpi.com/2073-4425/13/12/2323) and is part of the ongoing miRBind2 publication project, which focuses on improving miRNA target site prediction by training more accurate models on larger datasets. The aim is to better understand miRNA-mRNA interactions by making use of deep learning techniques to predict miRNA binding sites more effectively.

## Models Implemented
The repository contains the following models for miRNA target site classification:
- **Residual Network (ResNet)**
- **Bidirectional Long Short-Term Memory (BiLSTM)**
- **Deep Recurrent Neural Network (DeepRNN)**

## Project Tree
```bash
.
├── code
│   ├── BiLSTM_workflow.sh
│   ├── DeepRNN_workflow.sh
│   ├── ResNet_workflow.sh
│   ├── helper_functions
│   │   └── model_utils.py
│   └── machine_learning
│       ├── encode
│       │   ├── binding_2D_matrix_encoder.py
│       │   └── sequence_encoder_16ntPairs.py
│       ├── evaluate
│       │   ├── BiLSTM
│       │   │   ├── model_evaluate.py
│       │   │   └── model_predict.py
│       │   ├── DeepRNN
│       │   │   ├── model_evaluate.py
│       │   │   └── model_predict.py
│       │   └── ResNet
│       │       ├── model_evaluate.py
│       │       └── model_predict.py
│       └── train
│           ├── BiLSTM
│           │   ├── BiLSTM_Architectures.py
│           │   ├── BiLSTM_NoReg.py
│           │   └── BiLSTM_WithReg.py
│           ├── DeepRNN
│           │   ├── DeepRNN_Architectures.py
│           │   ├── DeepRNN_NoReg.py
│           │   └── DeepRNN_WithReg.py
│           └── ResNet
│               ├── ResNet_Architectures.py
│               ├── ResNet_NoReg.py
│               └── ResNet_WithReg.py
└── data
    ├── BiLSTM_data
    │   ├── testing
    │   │   └── README.md
    │   └── training
    │       └── README.md
    ├── DeepRNN_data
    │   ├── testing
    │   │   └── README.md
    │   └── training
    │       └── README.md
    └── ResNet_data
        ├── testing
        │   └── README.md
        └── training
            └── README.md
```
This project is organized into two main directories: `code` and `data`. The `code` directory contains the workflow scripts, helper functions and machine learning logic, which is further divided into the `encode`, `train` and `evaluate` directories. 

Each model architecture (BiLSTM, DeepRNN, and ResNet) has its own dedicated folder within both train and evaluate, allowing for clear traceability between files and their corresponding architectures. Furthermore, the helper_functions directory contains reusable utility functions that support model building, training, and evaluation across different architectures.

Encoding scripts handle sequence and binding matrix transformations, while workflow `.sh` scripts in the `root` streamline the execution process. Lastly, the `data` directory is divided by model type and workflow stage (training or testing), each with its own `README` file acting as placeholders to indicate which datasets should be placed there instead of the README file. This setup ensures modularity, clarity, and ease of use.

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
   python3 -m venv .venv                # create python environment
   source .venv/bin/activate            # activate environment
   python3 -m pip install --upgrade pip # upgrad pip to latest version
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Shell Scripts
Before running any shell script in the repository, you must grant execution permissions. For example, to set permissions for the `ResNet_workflow.sh` script in the ResNet model, run:
```bash
chmod +x code/ResNet_workflow.sh
```
To execute the script, use the file bath to the bash script along with its required user inputs:
```bash
code/ResNet_workflow.sh <user_input>
```

## Script Input Requirements
Each workflow script accepts a set of command-line arguments. In addition, you can use the -h or --help flag with any script to display the full usage information and a description of each input.

### ResNet_workflow.sh
**Usage:**
```bash
code/ResNet_workflow.sh [noncodingRNA | miRNA] [small | medium | large] [noreg | withreg] [plot_true | plot_false]
```

**Arguments:**

- `noncodingRNA | miRNA:`

  Specifies the column name for the microRNA in the datasets. (This accommodates datasets with different naming conventions.)

- `small | medium | large:`

  Selects the ResNet model variant based on its architecture parameters.

- `noreg | withreg:`

  Indicates whether to train the model without regularization (noreg) or with regularization (withreg).

- `plot_true | plot_false:`

  Specifies whether plots should be generated during training and evaluation.

### DeepRNN_workflow.sh and BiLSTM_workflow.sh

**Usage:**
```bash
code/DeepRNN_workflow.sh [noncodingRNA | miRNA] [noreg | withreg] [plot_true | plot_false]
```

```bash
code/BiLSTM_workflow.sh [noncodingRNA | miRNA] [noreg | withreg] [plot_true | plot_false]
```

**Arguments:**

- `noncodingRNA | miRNA:`

  Specifies the microRNA column name.

- `noreg | withreg:`

  Indicates the type of regularization to use.

- `plot_true | plot_false:`

  Determines whether to generate plots during the workflow.

>_**Note:** Please ensure that you supply the correct inputs corresponding to the desired operation, as mismatches may affect the naming conventions and subsequent mapping of models to plots and results._

## Datasets and Data Folder Placeholders
Within the repository, several data folders include placeholder `README.md` files. These placeholders indicate, by file name, which datasets need to be placed in each folder. 

**Note:** The file `AGO2_eCLIP_Manakov2022_full_dataset.tsv` should not be used.

### Dataset Sources
All required datasets can be obtained from the following link: [miRBench_Datasets - Zenodo](https://zenodo.org/records/14501607)

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
