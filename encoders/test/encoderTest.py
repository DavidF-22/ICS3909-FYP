# imports
import sys
sys.path.append('./')

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from encoder.sequence_encoder import get_label_dict, get_encoded_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# path to the dataset
training_file_path = 'datasets/training/Balanced_dataset.tsv'

# change column name if the dataset is CLASH2013 and plot names
if os.path.basename(training_file_path) == 'train_set_1_20_CLASH2013_paper.tsv':
    column_name = 'miRNA'
    dataset_name = 'CLASH_2013'
else:
    column_name = 'noncodingRNA'
    dataset_name = 'Balanced_dataset'

# read data
data = pd.read_csv(training_file_path, sep='\t')

# split data
training_data, _ = train_test_split(data, test_size=0.1, random_state=42)

# encode the testing data
print("----- <Encoding Training Data> -----")

# preprocess labels to avoid repeated DataFrame filtering
label_dict_training = get_label_dict(training_data, column_name)

# initialize lists to store encoded matrices and labels
encoded_training_data = []
training_labels = []

# encode training data
for _, row in training_data.iterrows():
    target_sequence = row['gene']
    mirna_sequence = row[column_name]
    
    encoded_matrix, label = get_encoded_matrix(label_dict_training , target_sequence, mirna_sequence)
    
    encoded_training_data.append(encoded_matrix)
    training_labels.append(label)
    
print("----- <Training Data Encoded Successfully> -----\n")

# convert to numpy array
# encoded_training_data = np.array(encoded_training_data)
# training_labels = np.array(training_labels)

# Print the first 3 encoded matrices
for i in range(3):
    print(f"Encoded Matrix {i+1}:")
    print(encoded_training_data[i])  # Print matrix
    print(f"Label: {training_labels[i]}\n")  # Print corresponding label
    
# Select the first encoded matrix (or any other index)
encoded_matrix_sample = np.array(encoded_training_data[0])

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(encoded_matrix_sample, cmap="viridis", annot=False)

# Titles and labels
plt.title("Encoded Sequence Heatmap")
plt.xlabel("Target Sequence (50 nucleotides)")
plt.ylabel("miRNA Sequence (25 nucleotides)")

# Show plot
plt.show()
