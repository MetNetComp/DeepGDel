import os
import pandas as pd

def read_gene_sequences(folder_path):  # Accept folder_path as argument
    gene_sequences = []
    gene_names = []

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            gene_names.append(str(df.iloc[:, 0].values[0]))
            gene_sequences.append(str(df.iloc[:, 1].values[0]))

    return gene_names, gene_sequences
