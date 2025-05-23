# imports
import sys
sys.path.insert(1, 'code/')

import os
import argparse
import pandas as pd
from tensorflow.keras.models import load_model
from machine_learning.train.ResNet.ResNet_Architectures import ResBlock
from helper_functions.model_utils import (set_seed,
                                          load_data, 
                                          make_files,
                                          simple_sort_key, 
                                          cleanup)

# * MAIN - PREDICT FUNCTION ---

# main pipeline
def main():
    # argument parser for dataset path and learning rate
    parser = argparse.ArgumentParser(description="Load trained model to make predictions for miRNA-mRNA target site classification")
    parser.add_argument("-rn_type", "--ResNet_type", required=True, type=str, help="Type of ResNet model to train (small [373,121], medium [1,360,001], large [16,691,073])")
    parser.add_argument("-e_data", "--encoded_data", required=True, type=str, help="List of paths to encoded test datasets (.npy files) used for predictions")
    parser.add_argument("-models", "--trained_models", required=True, type=str, help="List of paths to the trained models file (.keras or equivalent)")
    parser.add_argument("-reg", "--regularization", required=True, type=str, help="NoReg or WithReg using in naming the .tsv file")
    parser.add_argument("-s", "--seed", required=True, type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    # seeding
    set_seed(args.seed)
    
    # split model and dataset paths into lists and sort them
    test_data_files = sorted(args.encoded_data.split(','))
    model_files = sorted(args.trained_models.split(','), key=simple_sort_key)
    
    # check if --regularization is set to either "NoReg" or "WithReg"
    if args.regularization not in ["NoReg", "WithReg"]:
        raise ValueError(f"!!! Invalid regularization argument: {args.regularization} - Please use either 'NoReg' or 'WithReg' !!!")

    # initialise save predictions path
    save_dir = f"Saves_ResNet_{args.ResNet_type}_{args.regularization}/ResNet_Predictions"
    make_files(os.path.split(save_dir)[0], [os.path.split(save_dir)[1]])
    
    count = 1
    
    # iterate over all test data files and make predictions
    for test_data in test_data_files:
        # check if dataset file exists
        if not os.path.exists(test_data):
            print(f"!!! Error: File '{test_data}' not found! Skipping... !!!")
            continue

        # extract dataset name (remove directory and extension)
        dataset_name = os.path.splitext(os.path.basename(test_data))[0]
        dataset_name = dataset_name.replace('_test_dataset', '')
        
        # initialize dataframe to store model predictions
        predictions_df = pd.DataFrame()
        
        # load encoded test data
        print(f"\n----- <Loading encoded data from: {dataset_name}> -----\n")
        encoded_test_data, _ = load_data(data_file=test_data)

        # iterate over all model files
        for model_path in model_files:
            # get model name for column header
            model_name = os.path.basename(model_path)
            
            # check if model exists
            if not os.path.exists(model_path):
                print(f"!!! Error: Model '{model_path}' not found! Skipping... !!!")
                continue

            # load mdel using custom_objects to load the ResBlock class
            print(f"Loading model: {model_path} ...")
            model = load_model(model_path, custom_objects={'ResBlock': ResBlock})
            
            # get predictions
            predictions = model.predict(encoded_test_data).flatten()
            
            # store predictions in dataframe with model name as column
            predictions_df[model_name] = predictions
            
            del model, predictions
            cleanup()
            
        # define output path
        save_path = os.path.join(save_dir, f"{args.regularization}_{dataset_name}_pred{count}.tsv")
        
        count += 1

        # save predictions as .tsv file
        print(f"\nSaving predictions to: {save_path}")
        predictions_df.to_csv(save_path, sep='\t', index=False, float_format='%.6f')
        
        # clear memory-mapped data after each dataset
        del encoded_test_data, predictions_df
        cleanup()

    print(f"\n----- <All predictions saved successfully in {save_dir}> -----\n")


if __name__ == "__main__":
    main()
