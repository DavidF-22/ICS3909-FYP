import os
import pandas as pd

# define your dataset directory
dataset_dir = 'datasets'

# loop through each file in the dataset directory
for root, _, files in os.walk(dataset_dir):
    # determine the dataset type based on the folder name
    dataset_type = os.path.basename(root)  # get the name of the current directory

    # print the dataset type only if it's not the top-level dataset directory
    if dataset_type != 'datasets':  # check if the folder is not the main dataset directory
        print(f'\nDataset Type: {dataset_type.upper()}')

    for file in files:
        df = pd.read_csv(os.path.join(root, file), sep='\t')
        
        # count the number of 1's and 0's in the label column (assuming the label column is named 'label')
        count_of_ones = df['label'].value_counts().get(1, 0)
        count_of_zeros = df['label'].value_counts().get(0, 0)
        
        # print the counts for the current file with formatted output
        print(f'Dataset: {file:<45} Num of Ones: {count_of_ones:<10} Num of Zeros: {count_of_zeros:<10}')
