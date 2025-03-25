import pandas as pd

def read_metabolite_smiles(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Extract the first column (metabolite names)
    metabolite_names = df.iloc[:, 0].values
    
    # Extract the second column (SMILES features)
    smiles_features = df.iloc[:, 1].values
    
    # Return both arrays
    return metabolite_names, smiles_features