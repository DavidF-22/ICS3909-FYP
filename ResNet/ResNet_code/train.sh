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
    TRAIN_SCRIPT="ResNet_NoReg.py"
elif [ "$1" == "WithReg" ]; then
    TRAIN_SCRIPT="ResNet_WithReg.py"
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

# ensure script exists before attempting to execute it
SCRIPT_PATH="ResNet_code/machine_learning/train/$TRAIN_SCRIPT"
if [ ! -f "$SCRIPT_PATH" ]; then
    print_error "Training script '$SCRIPT_PATH' not found!"
    exit 1
fi

# output message if script is found and skip line
SCRIPT_BASENAME=$(basename "$SCRIPT_PATH")
print_success "Found Training Script: $TRAINING_SCRIPT_BASENAME"
print_success "Selected plot bool: $PLOT_BOOL"
print_success "Proceeding with execution..."
echo ""



# * training
# training_datasets.npy, training_labels.npy are required 
# (-plots and -lr [learning rate] are optional)

TRAINING_DATASET_PATH="datasets/training"

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
python3 "$SCRIPT_PATH" --encoded_data "$TRAIN_DATA_FILES" --encoded_labels "$TRAIN_LABEL_FILES" --plot_plots "$PLOT_BOOL"

# print success message
print_success "Training completed successfully"