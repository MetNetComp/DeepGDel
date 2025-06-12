import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def run_baseline(CBM='e_coli_core', M=100, relationship_folder_path=''):
    """
    Runs a baseline method on the gene-deletion dataset with balanced training data.
    
    Parameters:
    CBM (str): The model type ('e_coli_core', 'iMM904', or 'iML1515').
    M (int): Number of random baseline runs.
    """
    import os
    import numpy as np
    import pandas as pd

    all_accuracies, all_precisions, all_recalls, all_f1_scores, all_aucs = [], [], [], [], []
    last_train_0_count, last_train_1_count = 0, 0  # Save last training class counts

    for run_id in range(M):
        csv_files = [file for file in os.listdir(relationship_folder_path) if file.endswith('.csv')]
        num_train_files = int(0.2 * len(csv_files))
        train_files = np.random.choice(csv_files, size=num_train_files, replace=False)
        train_file_names = [os.path.splitext(file)[0] for file in train_files]
        remaining_file_names = [os.path.splitext(file)[0] for file in csv_files if file not in train_files]

        gene_count, gene_total, unique_genes = {}, {}, set()
        all_train_rows = []

        # Load and balance data within each gene
        for filename in train_files:
            df = pd.read_csv(os.path.join(relationship_folder_path, filename))
            for gene, group in df.groupby('Gene'):
                count_0 = (group['Deleted'] == 0).sum()
                count_1 = (group['Deleted'] == 1).sum()
                if count_0 == 0 or count_1 == 0:
                    all_train_rows.append(group)
                else:
                    min_count = min(count_0, count_1)
                    df_0 = group[group['Deleted'] == 0].sample(min_count, random_state=42)
                    df_1 = group[group['Deleted'] == 1].sample(min_count, random_state=42)
                    all_train_rows.append(pd.concat([df_0, df_1]))

        # Concatenate all and strictly balance global class distribution
        combined_train_df = pd.concat(all_train_rows)
        df_0_all = combined_train_df[combined_train_df['Deleted'] == 0]
        df_1_all = combined_train_df[combined_train_df['Deleted'] == 1]
        min_class_size = min(len(df_0_all), len(df_1_all))
        balanced_train_df = pd.concat([
            df_0_all.sample(min_class_size, random_state=42),
            df_1_all.sample(min_class_size, random_state=42)
        ])

        # Update last class distribution
        if run_id == M - 1:
            last_train_0_count = len(balanced_train_df[balanced_train_df['Deleted'] == 0])
            last_train_1_count = len(balanced_train_df[balanced_train_df['Deleted'] == 1])

        # Rebuild gene_count and gene_total
        gene_count, gene_total = {}, {}
        for _, row in balanced_train_df.iterrows():
            gene, deleted = row['Gene'], row['Deleted']
            if gene not in gene_count:
                gene_count[gene] = {'0': 0, '1': 0}
                gene_total[gene] = 0
            gene_count[gene][str(deleted)] += 1
            gene_total[gene] += 1
            unique_genes.add(gene)

        bound = 0.9
        genes_to_exclude = [
            gene for gene in gene_count
            if gene_count[gene]['0'] / gene_total[gene] >= bound or gene_count[gene]['1'] / gene_total[gene] >= bound
        ]

        accuracies, precisions, recalls, f1_scores, aucs = [], [], [], [], []
        for filename in remaining_file_names:
            df = pd.read_csv(os.path.join(relationship_folder_path, f"{filename}.csv"))
            filtered_df = df[~df['Gene'].isin(genes_to_exclude)]
            if filtered_df.empty:
                continue
            true_labels = filtered_df['Deleted'].values
            predicted_scores = filtered_df['Gene'].apply(
                lambda gene: gene_count[gene]['1'] / gene_total[gene] if gene in gene_count else 0.5
            ).values
            predicted_labels = (predicted_scores > 0.5).astype(int)

            TP = np.sum((predicted_labels == 1) & (true_labels == 1))
            TN = np.sum((predicted_labels == 0) & (true_labels == 0))
            FP = np.sum((predicted_labels == 1) & (true_labels == 0))
            FN = np.sum((predicted_labels == 0) & (true_labels == 1))

            accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100 if (TP + TN + FP + FN) > 0 else 0
            precision = (TP / (TP + FP)) * 100 if (TP + FP) > 0 else 0
            recall = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            try:
                auc = roc_auc_score(true_labels, predicted_scores) * 100
            except:
                auc = 0  # fallback in case only one class is present

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            aucs.append(auc)

        all_accuracies.append(np.mean(accuracies) if accuracies else 0)
        all_precisions.append(np.mean(precisions) if precisions else 0)
        all_recalls.append(np.mean(recalls) if recalls else 0)
        all_f1_scores.append(np.mean(f1_scores) if f1_scores else 0)
        all_aucs.append(np.mean(aucs) if aucs else 0)

    print("\n======== Balanced training dataset class distribution =======")
    print(f" Deleted (0)      = {last_train_0_count}")
    print(f" Non-deleted (1)  = {last_train_1_count}")
    print("\n====================== Baseline Report ======================")
    print(f"Overall Accuracy: {np.mean(all_accuracies):.2f}% ± {np.std(all_accuracies):.2f}")
    print(f"Macro-Averaged Precision: {np.mean(all_precisions):.2f}% ± {np.std(all_precisions):.2f}")
    print(f"Macro-Averaged Recall: {np.mean(all_recalls):.2f}% ± {np.std(all_recalls):.2f}")
    print(f"Macro-Averaged F1 Score: {np.mean(all_f1_scores):.2f}% ± {np.std(all_f1_scores):.2f}")
    print(f"Macro-Averaged AUC: {np.mean(all_aucs):.2f}% ± {np.std(all_aucs):.2f}")