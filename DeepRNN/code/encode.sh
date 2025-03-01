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
    echo "Usage: $0 [noncodingRNA | miRNA]"
    echo ""
    echo "Arguments:"
    echo "  [noncodingRNA | miRNA]   : Specify the micro-rna column name. Allowed values: noncodingRNA | miRNA"
    echo "                           : Note: This was done since datasets where enoucuntered with different column names for miRNA"
    echo ""
    echo "Options:"
    echo "  -h, --help                : Display usage information."
    exit 1
}

# display usage if help flag is passed
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi



# check if the script is run with required argument
if [ "$#" -ne 1 ]; then
    print_error "Invalid number of arguments. Use -h or --help for more information."
    exit 1
fi

# set the column name based on user input
if [ "$1" == "noncodingRNA" ]; then
    MIRNA_COL_NAME="noncodingRNA"
elif [ "$1" == "miRNA" ]; then
    MIRNA_COL_NAME="miRNA"
else
    print_error "Invalid argument: '$1' is not recognized. Allowed values: noncodingRNA | miRNA"
    exit 1
fi



# * encode

TRAINING_DATASETS_PATH="DeepRNN/data/DeepRNN_data/training"
TESTING_DATASETS_PATH="DeepRNN/data/DeepRNN_data/testing"
ENCODER_SCRIPT="DeepRNN/code/machine_learning/encode/sequence_encoder_16ntPairs.py"

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
    dataset_basename=$(basename "$dataset" ".tsv")

    python3 $ENCODER_SCRIPT --i_file $dataset --o_prefix $TRAINING_DATASETS_PATH/$dataset_basename --column_name "$MIRNA_COL_NAME"
    if [ $? -ne 0 ]; then
        print_error "Failed to encode training dataset '$dataset'!"
        exit 1
    fi
done

# encode testing datasets
for dataset in $TESTING_DATA_FILES; do
    # get basename of dataset
    dataset_basename=$(basename "$dataset" ".tsv")

    python3 $ENCODER_SCRIPT --i_file $dataset --o_prefix $TESTING_DATASETS_PATH/$dataset_basename --column_name "$MIRNA_COL_NAME"
    if [ $? -ne 0 ]; then
        print_error "Failed to encode testing dataset '$dataset'!"
        exit 1
    fi
done

echo ""
print_success "Successfully encoded all datasets"