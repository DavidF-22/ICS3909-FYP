# imports
import time
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras as k
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from encoder.binding_2D_matrix_encoder import binding_encoding
# from miRBench.encoder import miRBindEncoder

# Clean up resources to avoid OOM
from tensorflow.keras import backend as K
import gc
import tensorflow as tf

# * PARAMS ###############################################################################################################

# parameters
training_file_path = 'datasets/training/train_set_1_20_CLASH2013_paper.tsv'

alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1., "AU": 1., "UA": 1.}
input_shape = (50, 20, 1)  # shape of the input image
learning_rate = 0.001  # learning rate
epochs = 20  # number of epochs/dataset iterations
batch_size = 32  # batch size

# hyperparameter combinations
# reg_factors = [0.01, 0.005, 0.005, 0.01, 0.003, 0.002]
# dropout_rates = [0.5, 0.4, 0.5, 0.4, 0.3, 0.4]

reg_factors = [0.003, 0.002]
dropout_rates = [0.3, 0.4]

results_file_path = 'autoTrain_modelResults.txt'

testing_file_paths = [
    'datasets/testing/test_set_1_1_CLASH2013_paper.tsv',
    'datasets/testing/test_set_1_10_CLASH2013_paper.tsv',
    'datasets/testing/test_set_1_100_CLASH2013_paper.tsv',
]

regularizers = {
    # "L1" : L1, 
    "L2" : L2, 
    # "L1L2" : L1L2
}

elapsed_main_timers = []
    
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
        if reg_class is None:
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

# * HELPER FUNCTIONS ######################################################################################################

def write_to_file(message, mode='a', file_path=results_file_path):
    with open(file_path, mode) as results_file:
        results_file.write(message)
        
def clean_up():
    # Clear Keras session
    K.clear_session()

    # Explicitly delete objects
    del model
    del encoded_training_data, training_labels, encoded_validation_data, validation_labels

    # Force garbage collection
    gc.collect()

    # Reset TensorFlow graph
    tf.compat.v1.reset_default_graph()


# * MAIN ##################################################################################################################

def main():
    # clear output .txt file
    write_to_file("", 'w')

    # start main timer
    start_main_timer = time.time()

    # * LOAD AND ENCODE DATA ######################################################################################################

    # load the training dataset
    df_train = pd.read_csv(training_file_path, sep='\t')

    # Split df_train into actual training and validation sets
    training_data, validation_data = train_test_split(df_train, test_size=0.1, random_state=42)

    # encode the training and validation data
    encoded_training_data, training_labels = encode_dataset(training_data, "miRNA")
    encoded_validation_data, validation_labels = encode_dataset(validation_data, "miRNA")

    # build the ResNet model
    input_shape = encoded_training_data.shape[1:]  # assuming the encoded data is 4D (samples, height, width, channels)

    # * TRAINING AND TESTING ######################################################################################################

    # Loop through all regularizer types
    for regularizer_type in regularizers.keys():
        print(f"\nUsing Regularizer: {regularizer_type}")
        
        write_to_file(
            f"Using Regularizer: {regularizer_type}\n"
            "=" * 100 + "\n", 
            'a'
        )
                
        # Reset graph counter for each regularizer
        c = 1
        
        for reg_factor, dropout_rate in zip(reg_factors, dropout_rates):        
            print(f"\nTraining model with reg_factor={reg_factor}, dropout_rate={dropout_rate}\n")
            
            write_to_file(
                f"Training model with reg_factor={reg_factor}, dropout_rate={dropout_rate}\n"
                "=" * 100 + "\n\n", 
                'a'
            )

            # start training timer
            start_training_timer = time.time()

            # build model
            model = build_resnet(input_shape, reg_factor, dropout_rate, regularizer_type)

            # * TRAINING THE MODEL ######################################################################################################

            # train the model
            history = model.fit(encoded_training_data, 
                                training_labels, 
                                epochs=epochs,
                                verbose=1,
                                batch_size=batch_size, 
                                validation_data=(encoded_validation_data, validation_labels))

            # end training timer
            end_training_timer = time.time()
            # calculate and print the main time taken
            elapsed_training_timer = end_training_timer - start_training_timer
            print(f"\nTime taken for training and testing with reg_factor={reg_factor}, dropout_rate={dropout_rate}, regularizer={regularizer_type}: {round(elapsed_training_timer / 60, 2)} minutes\n")
            
            # * EVALUATING THE MODEL ######################################################################################################

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

            plt.savefig(f'CLASH_2013({c}_{regularizer_type}).png')
            plt.close()
            
            c += 1
            # * TESTING THE MODEL ######################################################################################################
            # evaluate the model on the testing datasets
            for i, testing_file_path in enumerate(testing_file_paths, start=1):
                print(f"----- <Evaluating Dataset {i}: {testing_file_path}> -----")
                
                write_to_file(f"**Dataset {i}:** {testing_file_path}\n", 'a')

                df_test = pd.read_csv(testing_file_path, sep='\t')
                encoded_testing_data, testing_labels = encode_dataset(df_test, "miRNA")
                
                test_loss, test_accuracy = model.evaluate(encoded_testing_data, testing_labels, verbose=0)
                
                predictions = model.predict(encoded_testing_data, verbose=0)
                roc_auc = roc_auc_score(testing_labels, predictions)
                
                write_to_file(
                    f"**Test loss:** {round(test_loss, 4)}\n"
                    f"**Test accuracy:** {round(test_accuracy, 4)} - {round(test_accuracy * 100, 2)}%\n"
                    f"**Area Under Curve:** {round(roc_auc, 4)}\n\n", 
                    'a'
                )

                print(f"Dataset {i} Results: Loss={round(test_loss, 4)}, Accuracy={round(test_accuracy, 4)}, ROC AUC={round(roc_auc, 4)}")
            
            
            
            # end main timer
            end_main_timer = time.time()
            # calculate main time taken
            elapsed_main_timer = end_main_timer - start_main_timer
            # store elapsed time
            elapsed_main_timers.append(elapsed_main_timer)
            # print main time taken
            print(f"Time taken for training and testing with reg_factor={reg_factor}, dropout_rate={dropout_rate}, regularizer={regularizer_type}: {round(elapsed_main_timer / 60, 2)} minutes\n")
            
            # write the time taken to the results file                
            write_to_file(
                f"**Time taken for training:** {round(elapsed_training_timer / 60, 2)} minutes\n"
                f"**Time taken for training and testing:** {round(elapsed_main_timer / 60, 2)} minutes\n\n"
                "=" * 100 + "\n",
                'a'
            )

        # * CLEAN UP RESOURCES ######################################################################################################
        clean_up()

    # calculate total time for all iterations
    total_time = sum(elapsed_main_timers)
    print(f"\nTotal time taken for all iterations: {round(total_time / 60, 2)} minutes")

    # write the total time to the results file
    write_to_file(
        f"\nTotal time taken for all iterations: {round(total_time / 60, 2)} minutes\n",
        'a'
    )

    print(f"\nResults saved to {results_file_path}. Graphs saved as 'CLASH(<#>_<regularizer_type>).png'.")



# * EXECUTION #############################################################################################################

# call main function
if __name__ == '__main__':
    main()