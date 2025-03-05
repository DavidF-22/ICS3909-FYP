# imports
import os
import gc
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# * PARAMS ---

# parameters
epochs = 20  # number of epochs/dataset iterations
batch_size = 32  # batch size
results_file_path = 'Saves/DeepRNN_training_NoReg_results.txt'

# define the directory where you want to save the model
save_dir = "Saves/DeepRNN_Models"

# hyperparameter combinations
dropout_rates = [0.05, 0.09, 0.13, 0.17, 0.21, 0.25]

regularizer_type = "NoReg"

# * BUILDING DeepRNN ---

# function to build the DeepRNN model with Attention layer and regularization
def DeepRNN(input_shape, dropout_rate, learning_rate):
    # clear any previous models
    tf.keras.backend.clear_session()
    
    print()
    
    # define input layer
    input_layer = layers.Input(shape=input_shape)
    print(f"Input layer shape: {input_layer.shape}")
    
    # Trainable weights for nucleotide pairs
    pair_embeddings = layers.Embedding(input_dim=16 + 1, output_dim=1)(input_layer)
    print(f"pair_embeddings shape before reshaping: {pair_embeddings.shape}")
    pair_embeddings = layers.Reshape((25,50))(pair_embeddings)
    print(f"pair_embeddings shape after reshaping: {pair_embeddings.shape}")

    # first RNN layer
    rnn1 = layers.SimpleRNN(units=128, return_sequences=True)(pair_embeddings)
    print(f"rnn1 shape: {rnn1.shape}")
    dropout1 = layers.Dropout(dropout_rate)(rnn1)
    print(f"dropout1 shape: {dropout1.shape}")

    # second RNN layer
    rnn2 = layers.SimpleRNN(units=128, return_sequences=True)(dropout1)
    print(f"rnn2 shape: {rnn2.shape}")
    dropout2 = layers.Dropout(dropout_rate)(rnn2)
    print(f"dropout2 shape: {dropout2.shape}")
    
    # third RNN layer
    rnn3 = layers.SimpleRNN(units=64, return_sequences=True)(dropout2)
    print(f"rnn3 shape: {rnn3.shape}")
    dropout3 = layers.Dropout(dropout_rate)(rnn3)
    print(f"dropout3 shape: {dropout3.shape}")

    # attention layer
    # attention = layers.Attention()([dropout3, dropout3])
    # print(f"attention shape: {attention.shape}")
    
    pooled = layers.GlobalAveragePooling1D()(dropout3)
    print(f"pooled shape: {pooled.shape}")

    # dense layer - fully connected hidden layer with 512 neurons
    dense = layers.Dense(units=512, activation='relu')(pooled)
    
    # batch normalization layer for stabilizing and accelerating the learning process
    batch_norm = layers.BatchNormalization()(dense)
    print(f"batch_norm shape: {batch_norm.shape}")
    dropout4 = layers.Dropout(dropout_rate)(batch_norm)
    print(f"dropout4 shape: {dropout4.shape}")

    # output layer for binary classification - 1 neuron (1 or 0)
    output = layers.Dense(units=1, activation='sigmoid')(dropout4)
    print(f"output shape: {output.shape}")

    # build model
    model = models.Model(inputs=input_layer, outputs=output)
    # compile model with Adam optimizer and binary crossentropy loss and accuracy metric
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    # Summary
    
    print()
    
    model.summary()
    
    print()
    
    return model

# * PLOTTING ---

# plot training and validation accuracy and loss
def plot_training(history, dataset_name, regularizer_type, save_dir, count_models, count_plots):
    # plotting training and validation accuracy and loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.axis(ymin=0.4, ymax=1)
    plt.title('DeepRNN - Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train_Accuracy', 'Validation_Accuracy'])
    plt.tight_layout()
    plt.grid()

    # plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('DeepRNN - Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train_Loss', 'Validation_Loss'])
    plt.tight_layout()
    plt.grid()

    plt.savefig(os.path.join(save_dir, f'DeepRNN_{dataset_name}_MultiTest_{regularizer_type}_{count_models}.{count_plots}.png'))
    plt.close('all')

# * LOADING DATA ---

def load_data(data_file, label_file):
    # load data
    encoded_data = np.load(data_file)
    encoded_labels = np.load(label_file)
    
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
    parser = argparse.ArgumentParser(description="Train a DeepRNN model for miRNA-mRNA target site classification")
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
            model = DeepRNN(input_shape, dropout_rate, learning_rate=args.learning_rate)
            
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
            model_path = os.path.join(save_dir, f"DeepRNN_multiTest_{regularizer_type}_{dataset_name}_{count_models}.keras")
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
