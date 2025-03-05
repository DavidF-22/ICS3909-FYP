# imports
import os
import gc
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from ResNet_Architectures import build_resnet_small, build_resnet_medium, build_resnet_large

# * PARAMS ---

# parameters
epochs = 20  # number of epochs/dataset iterations
batch_size = 32  # batch size
results_file_path = 'Saves/ResNet_training_NoReg_results.txt'

# define the directory where you want to save the model
save_dir = "Saves/ResNet_Models"

# hyperparameter combinations
dropout_rates = [0.05, 0.09, 0.13, 0.17, 0.21, 0.25]

regularizer_type = "NoReg"

# * PLOTTING ---

# plot training and validation accuracy and loss
def plot_training(history, dataset_name, regularizer_type, save_dir, count_models, count_plots):
    # plotting training and validation accuracy and loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.axis(ymin=0.4, ymax=1)
    plt.title('ResNet - Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train_Accuracy', 'Validation_Accuracy'])
    plt.tight_layout()
    plt.grid()

    # plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('ResNet - Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train_Loss', 'Validation_Loss'])
    plt.tight_layout()
    plt.grid()

    plt.savefig(os.path.join(save_dir, f'ResNet_{dataset_name}_MultiTest_{regularizer_type}_{count_models}.{count_plots}.png'))
    plt.close('all')

# * HANDELING MEMMAP ---   

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

def load_data(data_file, label_file):
    # load encoded data
    data_element_shape = (50, 20, 1)
    encoded_data_shape = get_memmap_shape(data_file, data_element_shape)
    encoded_data = np.memmap(data_file, dtype='float32', mode='r', shape=encoded_data_shape)

    # load labels (1D array)
    label_element_shape = (1,)  # labels are typically scalar per row
    label_shape = get_memmap_shape(label_file, label_element_shape)
    encoded_labels = np.memmap(label_file, dtype='float32', mode='r', shape=label_shape)
    
    return encoded_data, encoded_labels

# * CREATING DIRECTORY ---

# create directories for saving models and plots
def make_files(base_dir, sub_dirs):
    os.makedirs(base_dir, exist_ok=True)
    
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

# * MAIN - TRAINING FUNCTION ---

# main pipeline
def main():
    # argument parser for dataset path and learning rate
    parser = argparse.ArgumentParser(description="Train a ResNet model for miRNA-mRNA target site classification")
    parser.add_argument("-rn_type", "--ResNet_type", required=True, default=None, type=str, help="Type of ResNet model to train (small [373,121], medium [1,360,001], large [16,691,073])")
    parser.add_argument("-e_data", "--encoded_data", required=True, default=None, type=str, help="Path to the encoded training dataset (.npy file)")
    parser.add_argument("-e_labels", "--encoded_labels", required=True, default=None, type=str, help="Path to the encoded training labels (.npy file)")
    parser.add_argument("-plots", "--plot_plots", required=True, default=None, type=str, help="Wheather to save the training plots or not (true/false)")
    parser.add_argument("-lr", "--learning_rate", required=False, default=0.001, type=float, help="Learning rate for training")
    args = parser.parse_args()
    
    training_data_files = sorted(args.encoded_data.split(','), reverse=False)
    training_labels_files = sorted(args.encoded_labels.split(','), reverse=False)
    
    # loop through all datasets
    for training_data_file, training_label_file in zip(training_data_files, training_labels_files):
        # extract dataset name
        dataset_name = os.path.splitext(os.path.basename(training_data_file))[0]
        print(f"Dataset Name: {dataset_name}")

        # load the training dataset
        print(f"\n----- <Loading Encoded Training Data from {training_data_file}> -----")
        # load the encoded training data and labels        
        encoded_training_data, training_labels = load_data(training_data_file, training_label_file)
        
        print(f"Encoded data shape: {encoded_training_data.shape}")
        print(f"Training labels shape: {training_labels.shape}")
        print("----- <Encoded Training Data Loaded Successfully> -----\n")

        input_shape = encoded_training_data.shape[1:]
        print(f"Input shape: {input_shape}\n")
        
        # create the save directory
        make_files(os.path.split(save_dir)[0], [os.path.split(save_dir)[1]])
        
        # clear the results file
        with open(results_file_path, 'w') as results_file:
            pass
            
        # reset model count
        count_models = 1
        
        # loop through all hyperparameter combinations
        for dropout_rate in dropout_rates: 
            print(f"\nTraining model with {dataset_name}, dropout_rate={dropout_rate}\n")
            
            with open(results_file_path, 'a') as results_file:
                results_file.write(f"\nModel Number: {count_models}\n")
                results_file.write(f"Training model with {dataset_name}, dropout_rate={dropout_rate}")

            # start training timer
            start_training_timer = time.time()
            
            # build model
            if args.ResNet_type.lower() == "small":
                model = build_resnet_small(input_shape, dropout_rate, args.learning_rate)
            elif args.ResNet_type.lower() == "medium":
                model = build_resnet_medium(input_shape, dropout_rate, args.learning_rate)
            elif args.ResNet_type.lower() == "large":
                model = build_resnet_large(input_shape, dropout_rate, args.learning_rate)
            else:
                raise ValueError("!!! Invalid ResNet type. Only 'small', 'medium', or 'large' are recognised !!!")
            
            # train the model
            history = model.fit(encoded_training_data, 
                                training_labels, 
                                epochs=epochs,
                                batch_size=batch_size, 
                                validation_split=0.1,
                                verbose=1)

            # end training timer
            end_training_timer = time.time()
            
            # calculate and print the main time taken
            elapsed_training_timer = end_training_timer - start_training_timer
            print(f"\nTime taken for training with dropout_rate={dropout_rate}: {(elapsed_training_timer / 60):.3f} minutes\n")
            
            with open(results_file_path, 'a') as results_file:
                results_file.write(f"\nTime taken for training with dropout_rate={dropout_rate}: {(elapsed_training_timer / 60):.3f} minutes\n\n")
                results_file.write("=" * 100 + "\n")
            
            # ensure plot_flag is valid
            if args.plot_plots.lower() == "true":
                # plot training and validation accuracy and loss
                print("----- <Plotting training and validation accuracy and loss...> -----")
                plot_training(history, dataset_name, regularizer_type, save_dir, count_models, count_plots=1)
                
            elif args.plot_plots.lower() == "false":
                print("----- <Skipping plotting...> -----")
            else:
                raise ValueError("Invalid input for -pplots. Only 'true' or 'false' are allowed.")
                
            # save the model
            print("\n----- <Saving Model> -----")
            # construct the full file path
            model_path = os.path.join(save_dir, f"ResNet_multiTest_{regularizer_type}_{dataset_name}_{count_models}.keras")
            model.save(model_path)
            print("----- <Model Saved Successfully> -----\n\n")
            
            # increment model count
            count_models += 1
        
            # delete objects
            del model, history
            # force garbage collection
            gc.collect()
            # reset TensorFlow graph
            tf.keras.backend.clear_session()
            
        # delete objects
        del encoded_training_data, training_labels
        # force garbage collection
        gc.collect()

    # write completion message to results file
    with open(results_file_path, 'a') as results_file:
        results_file.write("\n--- Done ---")
    
    print(f"\nResults saved to {results_file_path}. Graphs saved as '<plot_type>_<dataset_name>_MultiTest_<regularizer_type>_<#>.png'")


if __name__ == "__main__":
    main()      
