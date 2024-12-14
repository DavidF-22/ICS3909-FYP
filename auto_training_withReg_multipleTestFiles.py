# imports
import time
import os

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras as k
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from encoder.binding_2D_matrix_encoder import binding_encoding
# from miRBench.encoder import miRBindEncoder

# Clean up resources to avoid OOM
import gc
import tensorflow as tf


# * PARAMS ###############################################################################################################


# parameters
training_file_paths = [
    'datasets/training/Balanced_dataset.tsv',
    # 'datasets/training/train_set_1_20_CLASH2013_paper.tsv'
]

alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1., "AU": 1., "UA": 1.}
input_shape = (50, 20, 1)  # shape of the input image
learning_rate = 0.001  # learning rate
epochs = 20  # number of epochs/dataset iterations
batch_size = 32  # batch size

# hyperparameter combinations
reg_factors = [0.01, 0.005, 0.005, 0.01, 0.003, 0.002]
dropout_rates = [0.05, 0.09, 0.13, 0.17, 0.21, 0.25]

plot_names = 'CLASH_2013'

results_file_path = 'autoTrain_modelResults.txt'

testing_file_paths = [
    'datasets/testing/test_set_1_1_CLASH2013_paper.tsv',
    'datasets/testing/test_set_1_10_CLASH2013_paper.tsv',
    'datasets/testing/test_set_1_100_CLASH2013_paper.tsv',
]

'''
Program stops running after completion of the first two regularisers. 
If there is not a fix split the process into two separate stages.

L1 and L2 then L1L2 separately.
'''
regularizers = {
    "L1" : L1, 
    "L2" : L2, 
    "L1L2" : L1L2
}


# * BUILDING RESNET ######################################################################################################


# defining a custom Keras layer which inturn implements a residual block
@register_keras_serializable()
class ResBlock(layers.Layer):
    """
    Defines a Residual block based on the original ResNet paper.
    The block either maintains the input dimensions or downsamples based on the specified parameters.
    """

    def __init__(self, reg_factor, regularizer_type, downsample=False, filters=16, kernel_size=3):
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
        
        # Dynamically select the regularizer
        reg_class = regularizers.get(regularizer_type)
        if not reg_class:
            raise ValueError(f"Unsupported regularizer type: {regularizer_type}")

        # initialize first convolution layer, with stride 1 or 2 depending on downsampling
        self.conv1 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=(1 if not self.downsample else 2),
                                   filters=self.filters,
                                   padding="same",
                                   kernel_regularizer=reg_class(reg_factor))
        self.activation1 = layers.ReLU()  # activation function after first convolution
        self.batch_norm1 = layers.BatchNormalization()  # batch normalization after first convolution
        
        # initialize second convolution layer with stride 1 (no downsampling here)
        self.conv2 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=1,
                                   filters=self.filters,
                                   padding="same",
                                   kernel_regularizer=reg_class(reg_factor))

        # third convolution if downsampling is needed to match input dimensions
        if self.downsample:
          self.conv3 = layers.Conv2D(kernel_size=1,
                                     strides=2,
                                     filters=self.filters,
                                     padding="same",
                                     kernel_regularizer=reg_class(reg_factor))

        self.activation2 = layers.ReLU()  # activation after second convolution
        self.batch_norm2 = layers.BatchNormalization()  # batch normalization after second convolution

    def call(self, inputs):
        """
        Forward pass for the residual block. Applies the convolutions, activation, and adds the skip connection.

        Parameters:
        - inputs: Input tensor

        Returns:
        - Tensor after applying the residual block transformation
        """
        # first convolution, activation, and batch normalization
        x = self.conv1(inputs)
        x = self.activation1(x)
        x = self.batch_norm1(x)
        
        # second convolution (no downsampling here)
        x = self.conv2(x)

        # adjust input dimensions if downsampling
        if self.downsample:
            inputs = self.conv3(inputs)

        # add the input (skip connection) to the output of the convolutions
        x = layers.Add()([inputs, x])

        # final activation and batch normalization
        x = self.activation2(x)
        x = self.batch_norm2(x)

        return x

    def get_config(self):
        """
        Returns the configuration of the residual block (required for saving and loading the model).
        """
        return {'filters': self.filters, 'downsample': self.downsample, 'kernel_size': self.kernel_size}
    
# define the ResNet model
def build_resnet(input_shape, reg_factor, dropout_rate, regularizer_type):
    """
    Builds a simple ResNet model using custom residual blocks.
    """
    # Dynamically select the regularizer
    reg_class = regularizers.get(regularizer_type)
    if reg_class is None:
        raise ValueError(f"Unsupported regularizer type: {regularizer_type}")
        
    inputs = layers.Input(shape=input_shape)

    # initial Conv Layer
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=reg_class(reg_factor))(inputs)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    # add ResBlocks
    x = ResBlock(reg_factor, regularizer_type, filters=64, downsample=False)(x)
    x = ResBlock(reg_factor, regularizer_type, filters=64, downsample=False)(x)

    # flatten and add dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=reg_class(reg_factor))(x)
    x = layers.Dropout(dropout_rate)(x)  # Dropout layer
    x = layers.Dense(1, activation='sigmoid', kernel_regularizer=reg_class(reg_factor))(x)  # binary classification (0 or 1)

    # build model
    model = models.Model(inputs, x)
    # compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    # # output model summary
    # model.summary()
    # print("\n")
    
    return model

# encode the data using your binding_2D_matrix_encoder's binding_encoding function
def encode_dataset(data, rna_type):
    # use the function from the binding_2D_matrix_encoder module
    return binding_encoding(data, rna_type, alphabet=alphabet)


# * PLOTTING ##############################################################################################################


def plot_training(history, count_plots, regularizer_type):
    # plotting training and validation accuracy and loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.axis(ymin=0.4, ymax=1)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train_Accuracy', 'Validation_Accuracy'])
    plt.tight_layout()
    plt.grid()

    # plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train_Loss', 'Validation_Loss'])
    plt.tight_layout()
    plt.grid()

    plt.savefig(f'training_{plot_names}({count_plots}_{regularizer_type}).png')
    plt.close()

def plot_roc_curve(testing_labels, predictions, roc_auc, count_plots, regularizer_type):
    # Plot ROC-AUC curve
    fpr, tpr, thresholds = roc_curve(testing_labels, predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig(f'ROC_{plot_names}({count_plots}_{regularizer_type}).png')
    plt.close()

def plot_pr_curve(testing_labels, predictions, count_plots, regularizer_type):
    precision, recall, thresholds = precision_recall_curve(testing_labels, predictions)
    pr_auc = auc(recall, precision)  # Compute the AUC for Precision-Recall Curve
            
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Precision-Recall Curve (AUC = {pr_auc:.4f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    plt.savefig(f'PR_{plot_names}({count_plots}_{regularizer_type}).png')
    plt.close()
    
    return pr_auc


# * MAIN ##################################################################################################################


def main():
    # clear output .txt file
    with open(results_file_path, 'w') as results_file:
        pass

    # start main timer
    start_main_timer = time.time()

    # * TRAINING AND TESTING ######################################################################################################
    for training_file_path in training_file_paths:
        # Loop through all regularizer types
        for regularizer_type in regularizers.keys():
            # Print regularizer type
            print(f"\nUsing Regularizer: {regularizer_type}")
            
            with open(results_file_path, 'a') as results_file:
                results_file.write(f"Using Regularizer: {regularizer_type}\n")
                results_file.write("=" * 100 + "\n")
                
            # reset elapsed main timers
            elapsed_main_timers = []
            # Reset graph counter for each regularizer
            count_plots = 1
            # Initialising column name - default to 'miRNA' - to account for different column names in different datasets
            column_name = 'noncodingRNA'
            
            # * LOAD AND ENCODE DATA ######################################################################################################

            # load the training dataset
            print("\n----- <Loading Training Datasets> -----")
            df_train = pd.read_csv(training_file_path, sep='\t')
            print("----- <Training Datasets Loaded Successfully> -----\n")

            # change column name if the dataset is CLASH2013 and plot names
            if os.path.basename(training_file_path) == 'train_set_1_20_CLASH2013_paper.tsv':
                column_name = 'miRNA'
                
            # encode the training and validation data
            print("----- <Encoding Training Datasets> -----")
            encoded_training_data, training_labels = encode_dataset(df_train, column_name)
            # encoded_validation_data, validation_labels = encode_dataset(validation_data, "noncodingRNA")
            print("----- <Training Datasets Encoded Successfully> -----\n")

            # get model input shape from encoded data
            input_shape = encoded_training_data.shape[1:]  # assuming the encoded data is 4D (samples, height, width, channels)
            
            
            
            # Loop through all hyperparameter combinations
            for reg_factor, dropout_rate in zip(reg_factors, dropout_rates):        
                print(f"\nTraining model with reg_factor={reg_factor}, dropout_rate={dropout_rate}\n")
                
                with open(results_file_path, 'a') as results_file:
                    results_file.write(f"Training model with reg_factor={reg_factor}, dropout_rate={dropout_rate}\n")
                    results_file.write("=" * 100 + "\n\n")

                # start training timer
                start_training_timer = time.time()

                # build model
                model = build_resnet(input_shape, reg_factor, dropout_rate, regularizer_type)

                # * TRAINING THE MODEL ######################################################################################################

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
                print(f"\nTime taken for training with reg_factor={reg_factor}, dropout_rate={dropout_rate}, regularizer={regularizer_type}: {round(elapsed_training_timer / 60, 2)} minutes\n")
                
                # plot training and validation accuracy and loss
                plot_training(history, count_plots, regularizer_type)
                
                # * TESTING THE MODEL ######################################################################################################
                
                # evaluate the model on the testing datasets
                for i, testing_file_path in enumerate(testing_file_paths, start=1):
                    print(f"----- <Evaluating Dataset {i}: {os.path.basename(testing_file_path)}> -----")
                    
                    with open(results_file_path, 'a') as results_file:
                        results_file.write(f"**Dataset {i}:** {os.path.basename(testing_file_path)}\n")

                    df_test = pd.read_csv(testing_file_path, sep='\t')
                    
                    encoded_testing_data, testing_labels = encode_dataset(df_test, 'miRNA')
                    
                    # validate input shape
                    # print(f"Expected input shape: {model.input_shape}")
                    # print(f"Encoded testing data shape: {encoded_testing_data.shape}")
                    
                    test_loss, test_accuracy = model.evaluate(encoded_testing_data, testing_labels, verbose=0)
                    
                    predictions = model.predict(encoded_testing_data, verbose=0)
                    roc_auc = roc_auc_score(testing_labels, predictions)
                    
                    # Plot ROC curve
                    plot_roc_curve(testing_labels, predictions, roc_auc, count_plots, regularizer_type)
                    
                    # Plot Precision-Recall curve
                    pr_auc = plot_pr_curve(testing_labels, predictions, count_plots, regularizer_type)
                    
                    count_plots += 1
                    
                    with open(results_file_path, 'a') as results_file:
                        results_file.write(f"**Test loss:** {round(test_loss, 4)}\n")
                        results_file.write(f"**Test accuracy:** {round(test_accuracy, 4)} - {round(test_accuracy * 100, 2)}%\n")
                        results_file.write(f"**ROC-AUC:** {round(roc_auc, 4)}\n")
                        results_file.write(f"**PR-AUC:** {round(pr_auc, 4)}\n\n")
                        
                    # print(f"Dataset {i} Results: Loss={round(test_loss, 4)}, Accuracy={round(test_accuracy, 4)}, AUC={round(roc_auc, 4)}")
                    print(f"Results: Test_Loss={round(test_loss, 4)}, Test_Accuracy={round(test_accuracy, 4)}, ROC-AUC={round(roc_auc, 4)}, PR-AUC={round(pr_auc, 4)}")
                    
                    
                    
                # end main timer
                end_main_timer = time.time()
                # calculate main time taken
                elapsed_main_timer = end_main_timer - start_main_timer
                # store elapsed time
                elapsed_main_timers.append(elapsed_main_timer)
                # print main time taken
                print(f"\nTime taken for training and testing with reg_factor={reg_factor}, dropout_rate={dropout_rate}, regularizer={regularizer_type}: {round(elapsed_main_timer / 60, 2)} minutes\n")
                
                # write the time taken to the results file   
                with open(results_file_path, 'a') as results_file:
                    results_file.write(f"**Time taken for training:** {round(elapsed_training_timer / 60, 2)} minutes\n")
                    results_file.write(f"**Time taken for training and testing:** {round(elapsed_main_timer / 60, 2)} minutes\n\n")
                    results_file.write("=" * 100 + "\n")             

            
                # * CLEAN UP RESOURCES ######################################################################################################
                
                
                # Explicitly delete objects
                del model, history
                del encoded_training_data, training_labels
                del encoded_testing_data, testing_labels, predictions
                del roc_auc, pr_auc

                # Force garbage collection
                gc.collect()

                # Reset TensorFlow graph
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                
            
            

            # calculate total time for all iterations
            total_time = sum(elapsed_main_timers)
            print(f"\nTotal time taken for all iterations: {round(total_time / 60, 2)} minutes")

            # write the total time to the results file
            with open(results_file_path, 'a') as results_file:
                results_file.write(f"\nTotal time taken for all iterations: {round(total_time / 60, 2)} minutes\n")

            print(f"\nResults saved to {results_file_path}. Graphs saved as '<plot_type>_{plot_names}(<#>_<regularizer_type>).png'.")



# * EXECUTION #############################################################################################################

# call main function
if __name__ == '__main__':
    main()