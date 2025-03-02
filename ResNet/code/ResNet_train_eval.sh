#!/bin/bash


# function to print error messages in red
print_error() {
    echo -e "\e[31mERROR: $1\e[0m"
}

# function to print success messages in green
print_success() {
    echo -e "\e[32m$1\e[0m"
}



# function to display usage information
usage() {
    echo "Usage: $0 [small | medium | large] [noreg | withreg] [plot_true | plot_false]"
    echo ""
    echo "Arguments:"
    echo "  small | medium | large    : Specify the ResNet type - [373,121] | [1,360,001] | [16,691,073] respectfully."
    echo "  noreg | withreg           : Specify the regularization type."
    echo "  plot_true | plot_false    : Specify whether to plot the results."
    echo ""
    echo "Options:"
    echo "  -h, --help                : Display usage information."
    exit 0
}

# display usage if help flag is passed
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi



# check if the script is run with two arguments
if [ "$#" -ne 3 ]; then
    print_error "Invalid number of arguments. Use -h or --help for more information."
    exit 1
fi

# set the ResNet type based on user input
if [ "$1" == "small" ]; then
    RESNET_TYPE="small"
elif [ "$1" == "medium" ]; then
    RESNET_TYPE="medium"
elif [ "$1" == "large" ]; then
    RESNET_TYPE="large"
else
    print_error "Invalid argument: '$1' is not recognized. Allowed values: small | medium | large"
    exit 1
fi

# set the training script based on user input
if [ "$2" == "noreg" ]; then
    REG_TYPE="NoReg"
    TRAIN_SCRIPT="ResNet_NoReg.py"
elif [ "$2" == "withreg" ]; then
    REG_TYPE="WithReg"
    TRAIN_SCRIPT="ResNet_WithReg.py"
else
    print_error "Invalid argument: '$2' is not recognized. Allowed values: noreg | withreg"
    exit 1
fi

# set the plot bool based on user input
if [ "$3" == "plot_true" ]; then
    PLOT_BOOL="true"
elif [ "$3" == "plot_false" ]; then
    PLOT_BOOL="false"
else
    print_error "Invalid argument: '$3' is not recognized. Allowed values: plot_true | plot_false"
    exit 1
fi



# * TRAIN ---
# training_datasets.npy, training_labels.npy are required 
# (-plots and -lr [learning rate] are optional)

# ensure script exists before attempting to execute it
SCRIPT_PATH="ResNet/code/machine_learning/train/ResNet/$TRAIN_SCRIPT"
if [ ! -f "$SCRIPT_PATH" ]; then
    print_error "Training script '$SCRIPT_PATH' not found!"
    exit 1
fi

# output message if script is found and skip line
SCRIPT_BASENAME=$(basename "$SCRIPT_PATH")
print_success "Found Training Script: $TRAINING_SCRIPT_BASENAME"
print_success "Selected ResNet type: $RESNET_TYPE"
print_success "Selected regularization type: $REG_TYPE"
print_success "Selected plot bool: $PLOT_BOOL"
print_success "Proceeding with execution..."
echo ""

TRAINING_DATASET_PATH="ResNet/data/ResNet_data/training"

# ensure training directory exist
if [ ! -d "$TRAINING_DATASET_PATH" ]; then
    print_error "Testing dataset directory '$TRAINING_DATASET_PATH' not found!"
    exit 1
fi

# get all training datasets containing '_dataset.npy'
TRAIN_DATA_FILES=$(find "$TRAINING_DATASET_PATH" -type f -name "*_dataset.npy" | sort | tr '\n' ',' | sed 's/,$//')
# get all training datasets containing '_label.npy'
TRAIN_LABEL_FILES=$(find "$TRAINING_DATASET_PATH" -type f -name "*_labels.npy" | sort | tr '\n' ',' | sed 's/,$//')

# validate that training dataset files were found
if [ -z "$TRAIN_DATA_FILES" ]; then
    print_error "No training datasets found in '$TRAINING_DATASET_PATH'!"
    exit 1
fi

# validate that training label files were found
if [ -z "$TRAIN_LABEL_FILES" ]; then
    print_error "No training labels found in '$TRAINING_DATASET_PATH'!"
    exit 1
fi

# print confirmation
print_success "Found training datasets"
print_success "Found training labels"
print_success "Training model..."
echo ""

# run the training script
python3 "$SCRIPT_PATH" --ResNet_type "$RESNET_TYPE" --encoded_data "$TRAIN_DATA_FILES" --encoded_labels "$TRAIN_LABEL_FILES" --plot_plots "$PLOT_BOOL"

# print success message
print_success "Training completed successfully"
echo ""



# * PREDICT ---
# models and testing_datasets.npy are needed

SAVED_MODELS_PATH="Saves/ResNet_Models"
PREDICTIONS_SCRIPT="ResNet/code/machine_learning/evaluate/model_predictions.py"

# ensure model dir exists
if [ ! -d "$SAVED_MODELS_PATH" ]; then
    print_error "Saved models directory '$SAVED_MODELS_PATH' not found!"
    exit 1
fi

# check if the prediction script exists
if [ ! -f "$PREDICTIONS_SCRIPT" ]; then
    print_error "Predictions script '$PREDICTIONS_SCRIPT' not found!"
    exit 1
fi

# get all testing datasets containing '_dataset.npy'
TEST_DATA_FILES=$(find "$TESTING_DATASET_PATH" -type f -name "*_dataset.npy" | sort | tr '\n' ',' | sed 's/,$//')
# get all trained models ending in '.keras'
MODEL_FILES=$(find "$SAVED_MODELS_PATH" -type f -name "*.keras" | sort | tr '\n' ',' | sed 's/,$//')

# validate that test dataset files were found
if [ -z "$TEST_DATA_FILES" ]; then
    print_error "No testing datasets found in '$TESTING_DATASET_PATH'!"
    exit 1
fi

# validate that model files were found
if [ -z "$MODEL_FILES" ]; then
    print_error "No trained models found in '$SAVED_MODELS_PATH'!"
    exit 1
fi

# print confirmation
PREDICTIONS_SCRIPT_BASENAME=$(basename "$PREDICTIONS_SCRIPT")
print_success "Found $PREDICTIONS_SCRIPT_BASENAME"
print_success "Selected regularization type: $REG_TYPE"
print_success "Found testing datasets"
print_success "Found trained models"
print_success "Running predictions..."
echo ""

# run the predictions script
python3 "$PREDICTIONS_SCRIPT" --ResNet_type "$RESNET_TYPE" --encoded_data "$TEST_DATA_FILES" --trained_models "$MODEL_FILES" --regularization "$REG_TYPE"

# output success message
print_success "Predictions obtained successfully"
echo ""



# * EVALUATE ---
# testing_datasets.npy, testing_labels.npy, -preds, -models are required
# (-plots is optional)

SAVED_PREDICTIONS_PATH="Saves/ResNet_Predictions"
EVALUATIONS_SCRIPT="ResNet/code/machine_learning/evaluate/model_evaluate.py"

# ensure predictions dir exists
if [ ! -d "$SAVED_PREDICTIONS_PATH" ]; then
    print_error "Saved models directory '$SAVED_PREDICTIONS_PATH' not found!"
    exit 1
fi

# check if the evaluations script exists
if [ ! -f "$EVALUATIONS_SCRIPT" ]; then
    print_error "Predictions script '$EVALUATIONS_SCRIPT' not found!"
    exit 1
fi

# get all testing datasets containing '_labels.npy'
TEST_LABEL_FILES=$(find "$TESTING_DATASET_PATH" -type f -name "*_labels.npy" | sort | tr '\n' ',' | sed 's/,$//')
# get all prediction files ending in '.tsv'
PREDICTION_FILES=$(find "$SAVED_PREDICTIONS_PATH" -type f -name "*.tsv" | sort | tr '\n' ',' | sed 's/,$//')

# validate that test label files were found
if [ -z "$TEST_LABEL_FILES" ]; then
    print_error "No testing datasets found in '$TESTING_DATASET_PATH'!"
    exit 1
fi

# validate that prediction files were found
if [ -z "$PREDICTION_FILES" ]; then
    print_error "No prediction files found in '$SAVED_PREDICTIONS_PATH'!"
    exit 1
fi

# print confirmation
EVALUATIONS_SCRIPT_BASENAME=$(basename "$EVALUATIONS_SCRIPT")
print_success "Found $EVALUATIONS_SCRIPT_BASENAME"
print_success "Selected regularization type: $REG_TYPE"
print_success "Selected plot bool: $PLOT_BOOL"
print_success "Found testing datasets"
print_success "Found testing labels"
print_success "Found predictions"
print_success "Found trained models"
print_success "Running evaluations..."
echo ""

# run the evaluations script
python3 "$EVALUATIONS_SCRIPT" --ResNet_type "$RESNET_TYPE" --encoded_data "$TEST_DATA_FILES" --encoded_labels "$TEST_LABEL_FILES" --predictions "$PREDICTION_FILES" --trained_models "$MODEL_FILES" --regularization "$REG_TYPE" --plot_plots "$PLOT_BOOL"

# output success message
echo ""
print_success "Evaluations obtained successfully"
echo ""
print_success "ResNet $RESNET_TYPE with $REG_TYPE pipeline completed successfully"