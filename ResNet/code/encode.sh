#!/bin/bash



# function to print error messages in red
print_error() {
    echo -e "\e[31mERROR: $1\e[0m"
}

# function to print success messages in green
print_success() {
    echo -e "\e[32m$1\e[0m"
}



# * encode

TRAINING_DATASETS_PATH="ResNet/data/training"
TESTING_DATASETS_PATH="ResNet/data/testing"
ENCODER_SCRIPT="code/machine_learning/encode/binding_2D_matrix_encoder.py"

# ensure training and testing directories exist
if [ ! -d "$TRAINING_DATASETS_PATH" ]; then
    print_error "Training dataset directory '$TRAINING_DATASETS_PATH' not found!"
    exit 1
fi

if [ ! -d "$TESTING_DATASETS_PATH" ]; then
    print_error "Testing dataset directory '$TESTING_DATASETS_PATH' not found!"
    exit 1
fi

# ensure encoder script exists
if [ ! -f "$ENCODER_SCRIPT" ]; then
    print_error "Encoder script '$ENCODER_SCRIPT' not found!"
    exit 1
fi

# get all .tsv files from training and testing directories
TRAINING_DATA_FILES=$(find "$TRAINING_DATASETS_PATH" -type f -name "*.tsv")
TESTING_DATA_FILES=$(find "$TESTING_DATASETS_PATH" -type f -name "*.tsv")

# validate that training datasets were found
if [ -z "$TRAINING_DATA_FILES" ]; then
    print_error "No training datasets found in '$TRAINING_DATASETS_PATH'!"
    exit 1
fi

# validate that testing datasets were found
if [ -z "$TESTING_DATA_FILES" ]; then
    print_error "No testing datasets found in '$TESTING_DATASETS_PATH'!"
    exit 1
fi

# print confirmation
ENCODER_SCRIPT_BASENAME=$(basename "$ENCODER_SCRIPT")
print_success "Found Encoder Script: $ENCODER_SCRIPT_BASENAME"
print_success "Found Training Datasets"
print_success "Found Testing Datasets"
print_success "Encoding..."
echo ""

# encode training datasets
for dataset in $TRAINING_DATA_FILES; do
    # get basename of dataset
    dataset_basename=$(basename "$dataset" "_paper.tsv")

    python3 $ENCODER_SCRIPT --i_file $dataset --o_prefix $TRAINING_DATASETS_PATH/$dataset_basename --column_name "miRNA"
    if [ $? -ne 0 ]; then
        print_error "Failed to encode training dataset '$dataset'!"
        exit 1
    fi
done

# encode testing datasets
for dataset in $TESTING_DATA_FILES; do
    # get basename of dataset
    dataset_basename=$(basename "$dataset" "_paper.tsv")

    python3 $ENCODER_SCRIPT --i_file $dataset --o_prefix $TESTING_DATASETS_PATH/$dataset_basename --column_name "miRNA"
    if [ $? -ne 0 ]; then
        print_error "Failed to encode testing dataset '$dataset'!"
        exit 1
    fi
done

echo ""
print_success "Successfully encoded all datasets"