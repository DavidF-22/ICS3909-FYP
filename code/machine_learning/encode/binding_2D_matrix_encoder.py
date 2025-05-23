import pandas as pd
import numpy as np
import argparse
import time


def binding_encoding(df, alphabet, column_name, tensor_dim=(50, 20, 1)):
    """
    Transform input sequence pairs to a binding matrix with corresponding labels.

    Parameters:
    - df: Pandas DataFrame with columns "noncodingRNA", "gene", "label"
    - alphabet: dictionary with letter tuples as keys and 1s when they bind
    - tensor_dim: 2D binding matrix shape

    Output:
    2D binding matrix, labels as np array
    """
    labels = df["label"].to_numpy()

    # Initialize dot matrix with zeros
    ohe_matrix_2d = np.zeros((len(df), *tensor_dim), dtype="float32")

    df = df.reset_index(drop=True)

    # Compile matrix with Watson-Crick interactions
    for index, row in df.iterrows():
        for bind_index, bind_nt in enumerate(row['gene'].upper()):
            for ncrna_index, ncrna_nt in enumerate(row[column_name].upper()):
                if ncrna_index >= tensor_dim[1]:
                    break
                base_pairs = bind_nt + ncrna_nt
                ohe_matrix_2d[index, bind_index, ncrna_index, 0] = alphabet.get(base_pairs, 0)

    return ohe_matrix_2d, labels


def encode_large_tsv_to_numpy(tsv_file_path, data_output_path, labels_output_path, column_name, chunk_size=10000):
    """
    Encode a large TSV file into a NumPy matrix using chunk processing.

    Parameters:
    - tsv_file_path: Path to the TSV file with dataset.
    - data_output_path: Path to the output data .npy file.
    - labels_output_path: Path to the output labels .npy file.
    - chunk_size: Number of rows to process at a time.
    """
    # Alphabet for Watson-Crick interactions
    alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1.}
    tensor_dim = (50, 20, 1)

    # Get total number of rows in the dataset
    num_rows = sum(len(df) for df in pd.read_csv(tsv_file_path, sep='\t', usecols=[0], chunksize=chunk_size))

    # Determine the shape of the output arrays
    labels_shape = (num_rows,)
    data_shape = (num_rows, *tensor_dim)

    # Create memory-mapped files
    ohe_matrix_2d = np.memmap(data_output_path, dtype='float32', mode='w+', shape=data_shape)
    labels = np.memmap(labels_output_path, dtype='float32', mode='w+', shape=labels_shape)

    row_offset = 0

    # Process each chunk
    for chunk in pd.read_csv(tsv_file_path, sep='\t', chunksize=chunk_size):
        encoded_data, encoded_labels = binding_encoding(chunk, alphabet, column_name, tensor_dim)

        # Write the chunk's data and labels to the memory-mapped files
        ohe_matrix_2d[row_offset:row_offset + len(chunk)] = encoded_data
        labels[row_offset:row_offset + len(chunk)] = encoded_labels
        row_offset += len(chunk)

    # Flush changes to disk
    ohe_matrix_2d.flush()
    labels.flush()


def main():
    """
    Based on "miRBind: A deep learning method for miRNA binding classification." Genes 13.12 (2022): 2323. https://doi.org/10.3390/genes13122323.
    Original implementation: https://github.com/ML-Bioinfo-CEITEC/miRBind

    Encodes miRNA and gene sequences into 2D-binding matrix.
    2D-binding matrix has shape (gene_max_len=50, miRNA_max_len=20, 1) and contains 1 for Watson-Crick interactions and 0 otherwise.
    """

    parser = argparse.ArgumentParser(
        description="Encode dataset to miRNA x target binding matrix. Outputs numpy file with matrices and and numpy file with corresponding labels. Expected columns of the dataset are 'noncodingRNA', 'gene' and 'label'")
    parser.add_argument('-i', '--i_file', type=str, required=True, help="Input dataset file name")
    parser.add_argument('-o', '--o_prefix', type=str, required=True, help="Output file name prefix")
    parser.add_argument('-col', '--column_name', type=str, required=True, help="Input miRNA column name")
    args = parser.parse_args()

    start = time.time()
    encode_large_tsv_to_numpy(args.i_file, args.o_prefix + '_dataset.npy', args.o_prefix + '_labels.npy', args.column_name)
    end = time.time()

    print("Elapsed time: ", end - start, " s.")


if __name__ == "__main__":
    main()