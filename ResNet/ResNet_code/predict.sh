#!/bin/bash


# function to print error messages in red
print_error() {
    echo -e "\e[31mERROR: $1\e[0m"
}

# function to print success messages in green
print_success() {
    echo -e "\e[32m$1\e[0m"
}



# check if the script is run with an argument
if [ "$#" -ne 1 ]; then
    print_error "Invalid number of arguments! - Usage: $0 [NoReg | WithReg]"
    exit 1
fi

# set the training script based on user input
if [ "$1" == "NoReg" ]; then
    REG_TYPE="NoReg"
elif [ "$1" == "WithReg" ]; then
    REG_TYPE="WithReg"
else
    print_error "Invalid argument: '$1' is not recognized. Allowed values: NoReg | WithReg"
    exit 1
fi



TESTING_DATASET_PATH="ResNet/datasets/testing"

# ensure testing directory exist
if [ ! -d "$TESTING_DATASET_PATH" ]; then
    print_error "Testing dataset directory '$TESTING_DATASET_PATH' not found!"
    exit 1
fi



# * predictions
# models and testing_datasets.npy are needed
SAVED_MODELS_PATH="Saves/ResNet_Models"
PREDICTIONS_SCRIPT="ResNet_code/machine_learning/evaluate/model_predictions.py"

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
python3 "$PREDICTIONS_SCRIPT" --encoded_data "$TEST_DATA_FILES" --trained_models "$MODEL_FILES" --regularization "$REG_TYPE"

# output success message
print_success "Predictions obtained successfully"