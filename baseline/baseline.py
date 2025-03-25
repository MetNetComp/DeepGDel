import os
import numpy as np
import pandas as pd

def run_baseline(CBM='e_coli_core', M=100, relationship_folder_path=''):
    """
    Runs a baseline method on the gene-deletion dataset.
    
    Parameters:
    CBM (str): The model type ('e_coli_core', 'iMM904', or 'iML1515').
    M (int): Number of random baseline runs.
    """
    
    all_accuracies, all_precisions, all_recalls, all_f1_scores = [], [], [], []
    
    for _ in range(M):
        csv_files = [file for file in os.listdir(relationship_folder_path) if file.endswith('.csv')]
        num_train_files = int(0.2 * len(csv_files))
        train_files = np.random.choice(csv_files, size=num_train_files, replace=False)
        train_file_names = [os.path.splitext(file)[0] for file in train_files]
        remaining_file_names = [os.path.splitext(file)[0] for file in csv_files if file not in train_files]
        
        gene_count, gene_total, unique_genes = {}, {}, set()
        
        for filename in train_files:
            df = pd.read_csv(os.path.join(relationship_folder_path, filename))
            for _, row in df.iterrows():
                gene, deleted = row['Gene'], row['Deleted']
                if gene not in gene_count:
                    gene_count[gene] = {'0': 0, '1': 0}
                    gene_total[gene] = 0
                gene_count[gene][str(deleted)] += 1
                gene_total[gene] += 1
                unique_genes.add(gene)
        
        bound = 0.9
        genes_to_exclude = [gene for gene in gene_count if gene_count[gene]['0'] / gene_total[gene] >= bound or gene_count[gene]['1'] / gene_total[gene] >= bound]
        
        accuracies, precisions, recalls, f1_scores = [], [], [], []
        
        for filename in remaining_file_names:
            df = pd.read_csv(os.path.join(relationship_folder_path, f"{filename}.csv"))
            filtered_df = df[~df['Gene'].isin(genes_to_exclude)]
            if filtered_df.empty:
                continue
            true_labels = filtered_df['Deleted'].values
            predicted_labels = filtered_df['Gene'].apply(lambda gene: 0 if gene_count[gene]['0'] / gene_total[gene] > 0.5 else 1).values
            
            TP, TN = np.sum((predicted_labels == 1) & (true_labels == 1)), np.sum((predicted_labels == 0) & (true_labels == 0))
            FP, FN = np.sum((predicted_labels == 1) & (true_labels == 0)), np.sum((predicted_labels == 0) & (true_labels == 1))
            
            accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100 if (TP + TN + FP + FN) > 0 else 0
            precision = (TP / (TP + FP)) * 100 if (TP + FP) > 0 else 0
            recall = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        all_accuracies.append(np.mean(accuracies) if accuracies else 0)
        all_precisions.append(np.mean(precisions) if precisions else 0)
        all_recalls.append(np.mean(recalls) if recalls else 0)
        all_f1_scores.append(np.mean(f1_scores) if f1_scores else 0)
    
    print("====================== Baseline Report ======================")
    print(f"Overall Accuracy: {np.mean(all_accuracies):.2f}% ± {np.std(all_accuracies):.2f}")
    print(f"Macro-Averaged Precision: {np.mean(all_precisions):.2f}% ± {np.std(all_precisions):.2f}")
    print(f"Macro-Averaged Recall: {np.mean(all_recalls):.2f}% ± {np.std(all_recalls):.2f}")
    print(f"Macro-Averaged F1 Score: {np.mean(all_f1_scores):.2f}% ± {np.std(all_f1_scores):.2f}")