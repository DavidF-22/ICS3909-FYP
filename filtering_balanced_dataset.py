import pandas as pd

def extract_columns(input_file, output_file, columns_to_keep):
    """
    Extract specific columns from a .tsv file and save them to another .tsv file.
    
    :param input_file: Path to the input .tsv file.
    :param output_file: Path to the output .tsv file.
    :param columns_to_keep: List of column names to extract.
    """
    try:
        # Load the .tsv file
        df = pd.read_csv(input_file, sep='\t')
        
        # Check if the required columns are in the file (fix: df.columns is a property, not a method)
        missing_columns = [col for col in columns_to_keep if col not in df.columns]
        if missing_columns:
            print(f"Error: The following columns are missing in the input file: {missing_columns}")
            return
        
        # Extract the required columns
        extracted_df = df[columns_to_keep]
        
        # Save to a new .tsv file
        extracted_df.to_csv(output_file, sep='\t', index=False)
        print(f"File saved successfully to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Input and output file paths
input_file_path = "datasets/Balanced_dataset.tsv"  # Replace with the path to your input file
output_file_path = "datasets/Filtered_Balanced_dataset.tsv"  # Replace with the desired output file path

# Columns to extract
columns_to_keep = ["noncodingRNA", "gene", "label"]

# Extract and save the columns
extract_columns(input_file_path, output_file_path, columns_to_keep)