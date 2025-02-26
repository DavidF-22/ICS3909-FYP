# imports
import os
import gc
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

# * MODEL ARCHITECTURE ---

# defining a custom Keras layer which inturn implements a residual block
@register_keras_serializable()
class ResBlock(layers.Layer):
    """
    Defines a Residual block based on the original ResNet paper.
    The block either maintains the input dimensions or downsamples based on the specified parameters.
    """

    def __init__(self, regularizer_type, downsample=False, filters=16, kernel_size=3):
        """
        Initializes the residual block with optional downsampling.
        
        Parameters:
        - downsample: Boolean, whether to downsample the input (using stride of 2)
        - filters: Number of filters for the Conv2D layers
        - kernel_size: Size of the convolution kernel
        """
        # calling the parent class constructor        
        super(ResBlock, self).__init__()
        
        # parameters for the residual block
        self.downsample = downsample
        self.filters = filters
        self.kernel_size = kernel_size
        self.regularizer_type = regularizer_type
        
        # first convolution: Conv -> BN -> ReLU
        self.conv1 = layers.Conv2D(filters=self.filters, 
                                   kernel_size=self.kernel_size, 
                                   strides=(1 if not self.downsample else 2), 
                                   padding="same")
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation("relu")
        
        # second convolution: Conv -> BN (activation applied after adding shortcut)
        self.conv2 = layers.Conv2D(filters=self.filters, 
                                   kernel_size=self.kernel_size, 
                                   strides=1, 
                                   padding="same")
        self.bn2 = layers.BatchNormalization()
        
        # if downsampling, adjust the shortcut branch with its own convolution and BN.
        if self.downsample:
            self.shortcut_conv = layers.Conv2D(filters=self.filters, 
                                               kernel_size=1, 
                                               strides=2, 
                                               padding="same")
            self.shortcut_bn = layers.BatchNormalization()
        else:
            self.shortcut_conv = None

    def call(self, inputs, training=False):
        """
        Forward pass for the residual block. Applies the convolutions, activation, and adds the skip connection.

        Parameters:
        - inputs: Input tensor

        Returns:
        - Tensor after applying the residual block transformation
        """
        # main branch: conv -> BN -> ReLU -> conv -> BN
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # shortcut branch
        shortcut = inputs
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)
        
        # add the shortcut and apply final activation
        x = layers.add([x, shortcut])

        return x

    def get_config(self):
        """
        Returns the configuration of the residual block (required for saving and loading the model).
        """
        reg_type = "NoReg"
        
        return {
            'filters': self.filters,
            'downsample': self.downsample,
            'kernel_size': self.kernel_size,
            'regularizer_type': reg_type
        }
    
    def build(self, input_shape):
        super(ResBlock, self).build(input_shape)
    
# * PLOTTING ---

# plot ROC curve for cross-validated model evaluation
def plot_roc_crossval(labels, predictions, save_dir, model_name, count_plots, n_splits=5):
    # Create StratifiedKFold instance (to keep class distribution balanced in each fold)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results for mean ROC curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(9, 7))
    
    # split labels into folds for cross-validation (uses precomputed predictions) - ensures balanced test sets for robust ROC-AUC evaluation
    for i, (train_idx, test_idx) in enumerate(cv.split(labels, labels)):
        # Get the test labels and the corresponding predictions
        y_test = labels[test_idx]      # True labels for the current fold
        y_pred = predictions[test_idx] # Corresponding predictions for this fold
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Interpolate TPR at common FPR points
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0  # Ensure TPR starts at 0
        
        # Plot individual fold ROC curve
        plt.plot(fpr, tpr, alpha=0.3, label=f"Fold {i+1} (AUC = {roc_auc:.3f})")

    # Compute mean and std deviation
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure last TPR is 1
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot mean ROC curve
    plt.plot(mean_fpr, mean_tpr, color="blue",
             label = fr"Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})",
             linewidth=2)

    # Plot chance level
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")

    # Shade standard deviation
    tprs_upper = np.minimum(mean_tpr + np.std(tprs, axis=0), 1)
    tprs_lower = np.maximum(mean_tpr - np.std(tprs, axis=0), 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color="blue", alpha=0.2)

    # Labels and legend
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Cross-Validated ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig(os.path.join(save_dir, f'ROC_{os.path.splitext(model_name)[0]}.{count_plots}.png'))
    plt.close('all')

# plot Precision-Recall curve for cross-validated model evaluation
def plot_pr_crossval(labels, predictions, save_dir, model_name, count_plots, n_splits=5):
    # Create StratifiedKFold instance
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results for mean PR curve
    precisions = []
    recalls = []
    aucs = []
    mean_recall = np.linspace(0, 1, 100)

    plt.figure(figsize=(9, 7))
    
    # Split labels into folds for cross-validation (uses precomputed predictions)
    for i, (train_idx, test_idx) in enumerate(cv.split(labels, labels)):
        # Get the test labels and corresponding predictions
        y_test = labels[test_idx]     
        y_pred = predictions[test_idx] 

        # Compute Precision-Recall curve and AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        pr_auc = auc(recall, precision)
        aucs.append(pr_auc)
        
        # Interpolate precision at common recall points
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))  # Reverse for interpolation
        precisions[-1][0] = 1.0  # Ensure precision starts at 1
        
        # Plot individual fold PR curve
        plt.plot(recall, precision, alpha=0.3, label=f"Fold {i+1} (AUC = {pr_auc:.3f})")

    # Compute mean and std deviation
    mean_precision = np.mean(precisions, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Plot mean PR curve
    plt.plot(mean_recall, mean_precision, color="blue",
             label = fr"Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})",
             linewidth=2)

    # Shade standard deviation
    precisions_upper = np.minimum(mean_precision + np.std(precisions, axis=0), 1)
    precisions_lower = np.maximum(mean_precision - np.std(precisions, axis=0), 0)
    plt.fill_between(mean_recall, precisions_lower, precisions_upper, color="blue", alpha=0.2)

    # Labels and legend
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Cross-Validated PR Curve - {model_name}")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    # Save plot
    plt.savefig(os.path.join(save_dir, f'PR_{os.path.splitext(model_name)[0]}.{count_plots}.png'))
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

# * MAIN - EVALUATION FUNCTION ---

def main():
    # argument parser for dataset path and learning rate
    parser = argparse.ArgumentParser(description="Load trained model to elvauate the performance of miRNA-mRNA target site classification")
    parser.add_argument("-e_data", "--encoded_data", required=True, default=None, type=str, help="List of path to the encoded testing dataset (.npy files)")
    parser.add_argument("-e_labels", "--encoded_labels", required=True, default=None, type=str, help="Lists of path to the encoded testing labels (.npy files)")
    parser.add_argument("-preds", "--predictions", required=True, default=None, type=str, help="List of paths to prediction files (.tsv files)")
    parser.add_argument("-models", "--trained_models", required=True, default=None, type=str, help="List of paths to the trained models (.keras or equivalent)")
    parser.add_argument("-reg", "--regularization", required=True, default=None, type=str, help="NoReg or WithReg using in naming the .tsv file")
    parser.add_argument("-plots", "--plot_plots", required=True, default=None, type=str, help="Wheather to save the training plots or not (true/false)")    
    args = parser.parse_args()
    
    # split model and dataset paths into lists and sort them
    test_data_files = sorted(args.encoded_data.split(','), reverse=True)
    test_label_files = sorted(args.encoded_labels.split(','), reverse=True)
    model_files = sorted(args.trained_models.split(','), key=simple_sort_key)
    prediction_files = sorted(args.predictions.split(','), reverse=True)
    
    # initialise save predictions path
    results_file_path = f'Saves/ResNet_evaluation_{args.regularization}_results.txt'
    
    if args.plot_plots == 'true':
        # create directory for saving plots
        save_dir = "Saves/ResNet_Evaluation"
        make_files(os.path.split(save_dir)[0], [os.path.split(save_dir)[1]])
        
    # clear the results file
    with open(results_file_path, 'w') as results_file:
        pass
    
    # iterate over all model files and load the model
    for model_path in model_files: 
        # get model name for easy identification
        model_name = os.path.basename(model_path)
        
        # load mdel
        print(f"\n----- <Loading model: {model_name}> -----\n")
        model = load_model(model_path, custom_objects={'ResBlock': ResBlock})

        
        with open(results_file_path, 'a') as results_file:
            results_file.write(f"Evaluating {model_name}\n")
            results_file.write("=" * 100 + "\n")
            
        count_plots = 2
            
        # iterate over all test data, test label and prediction files
        for test_data, test_label, prediction in zip(test_data_files, test_label_files, prediction_files):
            # get test data name for easy identification
            test_data_name = os.path.basename(test_data)
            
            # load the data and labels
            print(f"Loading encoded data from: {test_data} ...")
            test_data, test_labels = load_data(test_data, test_label)
            
            # load the predictions
            predictions_df = pd.read_csv(prediction, sep='\t')
            if model_name not in predictions_df.columns:
                print(f"!!! Warning: {model_name} not found in {prediction}. Available columns: {predictions_df.columns} !!!")
                continue
            predictions = predictions_df[model_name].values
            
            # evaluate the model and plot the ROC and PR curves
            test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=0)
            roc_auc = roc_auc_score(test_labels, predictions) # compute the AUC for ROC Curve
            
            precision, recall, thresholds = precision_recall_curve(test_labels, predictions)
            pr_auc = auc(recall, precision)  # compute the AUC for Precision-Recall Curve
            
            if args.plot_plots == 'true':
                print("Plotting ROC and PR curves ...")
                plot_roc_crossval(test_labels, predictions, save_dir, model_name, count_plots)
                plot_pr_crossval(test_labels, predictions, save_dir, model_name, count_plots)
            else:
                print("Skipping plotting ...")
            
            # write the results to the results file
            with open(results_file_path, 'a') as results_file:
                results_file.write(f"\nPlot Number: {count_plots}\n")
                results_file.write(f"Dataset: {test_data_name}\n")
                results_file.write(f"**Test loss:** {test_loss:.3f}\n")
                results_file.write(f"**Test accuracy:** {test_accuracy:.3f} - {(test_accuracy * 100):.3f}%\n")
                results_file.write(f"**PR-AUC:** {pr_auc:.3f}\n")
                results_file.write(f"**ROC-AUC:** {roc_auc:.3f}\n\n")
                results_file.write("=" * 100 + "\n")
            
            # clear the memory-mapped data
            del test_data, test_labels, predictions, predictions_df
            # force garbage collection
            gc.collect()
            # clear TensorFlow/Keras session to free memory
            tf.keras.backend.clear_session()
            
            count_plots += 1
            
    # write completion message to results file
    with open(results_file_path, 'a') as results_file:
        results_file.write("\n--- Done ---")

    
if __name__ == "__main__":
    main()