#!/bin/bash


# function to print error messages in red
print_error() {
    echo -e "\e[31mERROR: $1\e[0m"
}

# function to print success messages in green
print_success() {
    echo -e "\e[32m$1\e[0m"
}



# check if the script is run with two arguments
if [ "$#" -ne 2 ]; then
    print_error "Invalid number of arguments! - Usage: $0 [NoReg | WithReg] [plot_true | plot_false]"
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

# set the plot bool based on user input
if [ "$2" == "plot_true" ]; then
    PLOT_BOOL="true"
elif [ "$2" == "plot_false" ]; then
    PLOT_BOOL="false"
else
    print_error "Invalid argument: '$2' is not recognized. Allowed values: plot_true | plot_false"
    exit 1
fi



TESTING_DATASET_PATH="datasets/testing"

# ensure testing directory exist
if [ ! -d "$TESTING_DATASET_PATH" ]; then
    print_error "Testing dataset directory '$TESTING_DATASET_PATH' not found!"
    exit 1
fi



# * evaluate
# testing_datasets.npy, testing_labels.npy, -preds, -models are required
# (-plots is optional)
SAVED_MODELS_PATH="Saves/ResNet_Models"
SAVED_PREDICTIONS_PATH="Saves/ResNet_Predictions"
EVALUATIONS_SCRIPT="ResNet_code/machine_learning/evaluate/model_evaluate.py"

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

# get all testing datasets containing '_dataset.npy'
TEST_DATA_FILES=$(find "$TESTING_DATASET_PATH" -type f -name "*_dataset.npy" | sort | tr '\n' ',' | sed 's/,$//')
# get all testing datasets containing '_labels.npy'
TEST_LABEL_FILES=$(find "$TESTING_DATASET_PATH" -type f -name "*_labels.npy" | sort | tr '\n' ',' | sed 's/,$//')
# get all prediction files ending in '.tsv'
PREDICTION_FILES=$(find "$SAVED_PREDICTIONS_PATH" -type f -name "*.tsv" | sort | tr '\n' ',' | sed 's/,$//')
# get all trained models ending in '.keras'
MODEL_FILES=$(find "$SAVED_MODELS_PATH" -type f -name "*.keras" | sort | tr '\n' ',' | sed 's/,$//')

# validate that test dataset files were found
if [ -z "$TEST_DATA_FILES" ]; then
    print_error "No testing datasets found in '$TESTING_DATASET_PATH'!"
    exit 1
fi

if [ -z "$TEST_LABEL_FILES" ]; then
    print_error "No testing datasets found in '$TESTING_DATASET_PATH'!"
    exit 1
fi

# validate that prediction files were found
if [ -z "$PREDICTION_FILES" ]; then
    print_error "No prediction files found in '$SAVED_PREDICTIONS_PATH'!"
    exit 1
fi

# validate that model files were found
if [ -z "$MODEL_FILES" ]; then
    print_error "No trained models found in '$SAVED_MODELS_PATH'!"
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
python3 "$EVALUATIONS_SCRIPT" --encoded_data "$TEST_DATA_FILES" --encoded_labels "$TEST_LABEL_FILES" --predictions "$PREDICTION_FILES" --trained_models "$MODEL_FILES" --regularization "$REG_TYPE" --plot_plots "$PLOT_BOOL"

# output success message
echo ""
print_success "Evaluations pipeline completed successfully"