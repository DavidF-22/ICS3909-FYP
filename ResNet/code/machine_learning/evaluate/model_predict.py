# import
import os
import gc
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from ResNet.code.machine_learning.train.ResNet.ResNet_Architectures import ResBlock_SmallAndMedium, ResBlock_Large
  
# * PLOTTING ---

# get the shape of the memory-mapped file (dataset) - helps in loading the data
def get_memmap_shape(file_path, element_shape, dtype=np.float32):
    """Infers the first dimension (dataset_size) for a memory-mapped file."""
    # size of one element in bytes
    item_size = np.prod(element_shape) * np.dtype(dtype).itemsize
    # file size in bytes
    total_size = os.path.getsize(file_path)
    # number of elements
    dataset_size = total_size // item_size
    
    # print(f"Total size: {dataset_size}")
    
    # return the shape tuple
    return (dataset_size, *element_shape)

# * LOADING DATA ---

def load_data(data_file):
    # load encoded data
    data_element_shape = (50, 20, 1)
    encoded_data_shape = get_memmap_shape(data_file, data_element_shape)
    encoded_data = np.memmap(data_file, dtype='float32', mode='r', shape=encoded_data_shape)
    
    return encoded_data

# sorting
def simple_sort_key(path):
    if "L1" in path and "L1L2" not in path:
        return 0  # L1
    elif "L1L2" in path:
        return 1  # L1L2
    elif "L2" in path:
        return 2  # L2
    else:
        return 3

# * CREATING DIRECTORY ---

# create directories for saving models and plots
def make_files(base_dir, sub_dirs):
    os.makedirs(base_dir, exist_ok=True)
    
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

# * MAIN ---

# main pipeline
def main():
    # argument parser for dataset path and learning rate
    parser = argparse.ArgumentParser(description="Load trained model to make predictions for miRNA-mRNA target site classification")
    parser.add_argument("-rn_type", "--ResNet_type", required=True, default=None, type=str, help="Type of ResNet model to train (small [373,121], medium [1,360,001], large [16,691,073])")
    parser.add_argument("-e_data", "--encoded_data", required=True, default=None, type=str, help="List of paths to encoded test datasets (.npy files) used for predictions")
    parser.add_argument("-models", "--trained_models", required=True, default=None, type=str, help="List of paths to the trained models file (.keras or equivalent)")
    parser.add_argument("-reg", "--regularization", required=True, default="NoReg", type=str, help="NoReg or WithReg using in naming the .tsv file")
    args = parser.parse_args()
    
    # split model and dataset paths into lists and sort them
    test_data_files = sorted(args.encoded_data.split(','), reverse=True)
    model_files = sorted(args.trained_models.split(','), key=simple_sort_key)
    
    # check if --regularization is set to either "NoReg" or "WithReg"
    if args.regularization not in ["NoReg", "WithReg"]:
        raise ValueError(f"!!! Invalid regularization argument: {args.regularization} - Please use either 'NoReg' or 'WithReg' !!!")

    # initialise save predictions path
    save_dir = "Saves/ResNet_Predictions"
    make_files(os.path.split(save_dir)[0], [os.path.split(save_dir)[1]])
    
    # select the ResBlock class based on the ResNet type
    if args.ResNet_type == 'small' or args.ResNet_type == 'medium':
        RESBLOCK_CLASS = ResBlock_SmallAndMedium
    elif args.ResNet_type == 'large':
        RESBLOCK_CLASS = ResBlock_Large
    
    # iterate over all test data files and make predictions
    for test_data in test_data_files:
        # check if dataset file exists
        if not os.path.exists(test_data):
            print(f"!!! Error: File '{test_data}' not found! Skipping... !!!")
            continue

        # extract dataset name (remove directory and extension)
        dataset_name = os.path.splitext(os.path.basename(test_data))[0]
        
        # initialize dataframe to store model predictions
        predictions_df = pd.DataFrame()
        
        # load encoded test data
        print(f"\n----- <Loading encoded data from: {test_data}> -----\n")
        encoded_test_data = load_data(test_data)

        # iterate over all model files
        for model_path in model_files:
            # check if model exists
            if not os.path.exists(model_path):
                print(f"!!! Error: Model '{model_path}' not found! Skipping... !!!")
                continue
            
            # get model name for column header
            model_name = os.path.basename(model_path)

            # load mdel using custom_objects to load the ResBlock class
            print(f"Loading model: {model_path} ...")
            model = load_model(model_path, custom_objects={'ResBlock': RESBLOCK_CLASS})
            
            # get predictions
            predictions = model.predict(encoded_test_data).flatten()
            
            # store predictions in dataframe with model name as column
            predictions_df[model_name] = predictions
            
            # free TensorFlow session & memory after using the model
            tf.keras.backend.clear_session()
            # clear model from memory
            del model, predictions
            gc.collect()
            
        # define output path
        save_path = os.path.join(save_dir, f"ResNet_{args.regularization}_{dataset_name}.tsv")

        # save predictions as .tsv file
        print(f"\nSaving predictions to: {save_path}")
        predictions_df.to_csv(save_path, sep='\t', index=False, float_format='%.6f')

        print(f"Predictions for {test_data} saved to {save_path}\n")
        
        # clear memory-mapped data after each dataset
        del encoded_test_data, predictions_df
        gc.collect()

    print(f"\n----- <All predictions saved successfully in {save_dir}> -----\n\n")

if __name__ == "__main__":
    main()
