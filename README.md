# miRBind2: miRNA Target Site Prediction

## Project Overview
miRBind is a deep learning-based project focused on predicting microRNA (miRNA) target site binding. The project explores different neural network architectures to enhance the accuracy of miRNA-mRNA interaction predictions. This work is part of ongoing research in bioinformatics, specifically in the area of post-transcriptional gene regulation.

## Models Implemented
The repository contains the following models for miRNA target site classification:
- **Convolutional Neural Network (CNN)**
- **Residual Network (ResNet)**
- **Bidirectional Long Short-Term Memory (BiLSTM) with Attention**
- **Long Short-Term Memory (LSTM)**

## Cloning the Repository
To clone this repository, use the following command:
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

## Setting Up the Environment
1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Models
Each model is implemented in a separate script. Run them as follows:
```bash
python train_resnet.py   # Example for training ResNet model
python train_bilstm.py   # Example for training BiLSTM model
```
Modify the scripts as needed for different dataset configurations and hyperparameters.

## Contact
For any questions or contributions, feel free to reach out or create an issue in the repository.

---
This project is part of ongoing research in miRNA target site prediction and aims to improve the understanding of miRNA-mRNA interactions using deep learning techniques.
