import numpy as np

# nucleotide pairs and their corresponding indices
nucleotide_pairs = [
    ('A', 'A'), ('A', 'T'), ('A', 'C'), ('A', 'G'),
    ('T', 'A'), ('T', 'T'), ('T', 'C'), ('T', 'G'),
    ('C', 'A'), ('C', 'T'), ('C', 'C'), ('C', 'G'),
    ('G', 'A'), ('G', 'T'), ('G', 'C'), ('G', 'G')
]

# create a dictionary to map nucleotide pairs to indices
pair_to_index = {pair: i for i, pair in enumerate(nucleotide_pairs)}
num_pairs = len(nucleotide_pairs)

# Helper function to pad or trim sequences to the desired length
def pad_or_trim(seq, desired_length):
    # if sequence length is greater than the desired length
    if len(seq) > desired_length:
        # trim the sequence to the desired length
        return seq[:desired_length]
    else:
        # pad the sequence with 'N' to the desired length
        return seq + 'N' * (desired_length - len(seq))

# * ---

# Encoding function to create a complementarity matrix with progress logging
def encode_complementarity(target_seq, mirna_seq, miRNA_length=25, target_length=50):
    """
    Encodes the complementarity between target and miRNA sequences into a matrix.
    Logs progress to a file.

    Parameters:
    - target_seq: Target RNA sequence (string).
    - mirna_seq: miRNA sequence (string).
    - miRNA_length: Desired length of the miRNA sequence.
    - target_length: Desired length of the target RNA sequence.
    - log_file: Path to the file where progress will be logged.

    Returns:
    - A 2D matrix (miRNA_length x target_length) with encoded nucleotide pair indices.
    """
    # initialize the encoding matrix
    arr = np.zeros((miRNA_length, target_length), dtype=np.int32)

    for i in range(miRNA_length):
        for j in range(target_length):
            if i < len(mirna_seq) and j < len(target_seq):
                if mirna_seq[i] == 'N' or target_seq[j] == 'N':
                    arr[i, j] = num_pairs  # Special index for 'N'
                else:
                    pair = (mirna_seq[i], target_seq[j])
                    arr[i, j] = pair_to_index.get(pair, num_pairs)

    return arr

# Function to preprocess the labels and store them in a dictionary
def get_label_dict(df, miRNA_column):
    # initialize empty dictionary
    label_dict = {}

    # iterate through each row of the DataFrame
    for _, row in df.iterrows():
        
        # extract the values from the rowr
        miRNA = row[miRNA_column]
        gene = row["gene"]
        label = row["label"]

        # store label using (miRNA/noncodingRNA, gene) as the key
        label_dict[(miRNA, gene)] = label

    # return label dictionary
    return label_dict

# Function to get the encoded matrix and label for a given dataset and input sequences
def get_encoded_matrix(label_dict, target_sequence, mirna_sequence):
    key = (mirna_sequence, target_sequence)
    if key not in label_dict:
        raise ValueError("No matching label found for the given sequences.")
    
    # extract the label for the specific sequences
    label = label_dict[key]
    
    # pad or trim sequences to the desired lengths
    target_sequence = pad_or_trim(target_sequence, desired_length=50)
    mirna_sequence = pad_or_trim(mirna_sequence, desired_length=25)

    # encode the sequences
    encoded_matrix = encode_complementarity(target_sequence, mirna_sequence)
    
    # return encoded_matrix and label
    return encoded_matrix, label