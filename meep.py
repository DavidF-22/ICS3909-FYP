import sys
sys.path.append('../')

# imports
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras as k
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.reguliarizers import l1, l2, l1_l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from encoder.binding_2D_matrix_encoder import binding_encoding
# from miRBench.encoder import miRBindEncoder



# parameters
training_file_path = '../datasets/Balanced_dataset.tsv'

alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1., "AU": 1., "UA": 1.}
input_shape = (50, 20, 1)  # shape of the input image
learning_rate = 0.001  # learning rate
epochs = 20  # number of epochs/dataset iterations
batch_size = 32  # batch size

# defining a custom Keras layer which inturn implements a residual block
@register_keras_serializable()
class ResBlock(layers.Layer):
    """
    Defines a Residual block based on the original ResNet paper.
    The block either maintains the input dimensions or downsamples based on the specified parameters.
    """

    def __init__(self, downsample=False, filters=16, kernel_size=3):
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

        # initialize first convolution layer, with stride 1 or 2 depending on downsampling
        self.conv1 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=(1 if not self.downsample else 2),
                                   filters=self.filters,
                                   padding="same")
        self.activation1 = layers.ReLU()  # activation function after first convolution
        self.batch_norm1 = layers.BatchNormalization()  # batch normalization after first convolution
        
        # initialize second convolution layer with stride 1 (no downsampling here)
        self.conv2 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=1,
                                   filters=self.filters,
                                   padding="same")

        # third convolution if downsampling is needed to match input dimensions
        if self.downsample:
          self.conv3 = layers.Conv2D(kernel_size=1,
                                     strides=2,
                                     filters=self.filters,
                                     padding="same")

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
def build_resnet(input_shape):
    """
    Builds a simple ResNet model using custom residual blocks.
    """
    inputs = layers.Input(shape=input_shape)

    # initial Conv Layer
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    # add ResBlocks
    x = ResBlock(filters=64, downsample=False)(x)
    x = ResBlock(filters=64, downsample=False)(x)

    # flatten and add dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)  # binary classification (0 or 1)

    # build model
    model = models.Model(inputs, x)
    # compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    # output model summary
    model.summary()
    
    return model

# encode the data using your binding_2D_matrix_encoder's binding_encoding function
def encode_dataset(data, rna_type):
    # use the function from the binding_2D_matrix_encoder module
    return binding_encoding(data, rna_type, alphabet=alphabet)

# load the training dataset
print("----- <Loading Training Dataset> -----")
df_train = pd.read_csv(training_file_path, sep='\t')
print("----- <Dataset Loaded Successfully> -----\n")

# print the dataset shape and first few rows
print(f"Training Dataset shape: {df_train.shape}")
print(f"First few rows of the dataset:\n{df_train.head()}\n")

# Split df_train into actual training and validation sets
training_data, validation_data = train_test_split(df_train, test_size=0.1, random_state=42)
print(f"Size of training set: {len(training_data)}")
print(f"Size of validation set: {len(validation_data)}\n")

print("----- <Encoding Data> -----")
encoded_training_data, training_labels = encode_dataset(training_data, "noncodingRNA")
print("----- <Training Data Encoded Successfully> -----")
encoded_validation_data, validation_labels = encode_dataset(validation_data, "noncodingRNA")
print("----- <Validation Data Encoded Successfully> -----")
print("----- <Successfully Encoded Data> -----\n")
'''Displaying the encoded data'''
# print(f"Encoded training data shape: {encoded_training_data.shape}")
# print(f"Encoded validation data shape: {encoded_validation_data.shape}\n")
# print(f"Encoded testing data shape: {encoded_testing_data.shape}")
# print(f"First encoded training example:\n{encoded_training_data[0]}")
# print(f"First training label: {training_labels[0]}\n")
# print(f"First encoded testing example:\n{encoded_testing_data[0]}")

# build the ResNet model
input_shape = encoded_training_data.shape[1:]  # assuming the encoded data is 4D (samples, height, width, channels)
print("----- <Building Model> -----")
model = build_resnet(input_shape)
print("----- <Model Built Successfully> -----\n")

# train the model
print("----- <Training Model> -----")
history = model.fit(encoded_training_data, training_labels, epochs=epochs, batch_size=batch_size, 
                    validation_data=(encoded_validation_data, validation_labels))
print("----- <Model Trained Successfully> -----\n")

# evaluate the model

# parameters
testing_file_path = '../datasets/testing/test_set_1_100_CLASH2013_paper.tsv'

# load the testing dataset
print("----- <Loading Testing Dataset> -----")
df_test = pd.read_csv(testing_file_path, sep='\t')
print("----- <Dataset Loaded Successfully> -----\n")

# print the dataset shape and first few rows
print(f"Testing Dataset shape: {df_test.shape}")
print(f"First few rows of the dataset:\n{df_test.head()}\n")

# encode the testing data
print("----- <Encoding Data> -----")
encoded_testing_data, testing_labels = encode_dataset(df_test, "miRNA")         # encode_dataset(df_test, "noncodingRNA")
print("----- <Testing Data Encoded Successfully> -----")
print("----- <Successfully Encoded Data> -----\n")


print("----- <Evaluating Model> -----")
test_loss, test_accuracy = model.evaluate(encoded_testing_data, testing_labels)

# get predictions
predictions = model.predict(encoded_testing_data)
roc_auc = roc_auc_score(testing_labels, predictions)
print("----- <Model Evaluated Successfully> -----\n")

# display results
print(f"Test loss: {round(test_loss, 4)}\nTest accuracy: {round(test_accuracy, 4)} - {round(test_accuracy, 4) * 100}%")
print(f"Area Under Curve: {round(roc_auc, 4)}")


# save the model
print("----- <Saving Model> -----")
model.save("miRBind_ResNet.keras")
print("----- <Model Saved Successfully> -----\n")