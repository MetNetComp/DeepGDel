import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import time
import os

def predict_gene_deletions_for_metabolite(model, metabolite_name, metabolite_names, gene_sequences, smiles_features, relationships, device, genes_to_exclude):
    # Find the index of the metabolite name
    meta_index = metabolite_names.tolist().index(metabolite_name)
    
    # Get the corresponding fingerprint feature
    fingerprint_feature = smiles_features[meta_index]  # Assuming smiles_features is now a NumPy array
    
    # Prepare lists to store predictions, probabilities, and true labels
    predicted_deletions = []
    true_labels = []
    gene_names = []
    
    # Model prediction for each gene sequence
    model.eval()
    with torch.no_grad():
        # Convert the fingerprint feature to a tensor
        fingerprint_tensor = torch.tensor(fingerprint_feature, dtype=torch.float32).to(device)
        
        # Prepare gene sequences tensor
        gene_seqs_tensor = torch.tensor(gene_sequences, dtype=torch.long).to(device)
        
        # Repeat or expand fingerprint tensor for each gene sequence
        fingerprint_feat = fingerprint_tensor.expand(len(gene_sequences), -1).contiguous()
        
        # Make predictions
        combined_input = (gene_seqs_tensor, fingerprint_feat)
        output = model(*combined_input)  # Assuming the model returns a tuple
        prediction = output[0]  # Extract the classification output from the tuple
        predicted_probs = torch.sigmoid(prediction).squeeze().cpu().numpy()
        
        # Convert probabilities to binary predictions
        predicted_deletions = (predicted_probs >= 0.6).astype(int)
        
        # Get true labels from relationships data
        if metabolite_name in relationships:
            gene_names, true_labels = relationships[metabolite_name]
            true_labels = np.array(true_labels)
        else:
            print(f"No relationship data found for metabolite '{metabolite_name}'")
            return [], predicted_deletions, predicted_probs, []

        # Filter out excluded genes
        filtered_indices = [i for i, gene in enumerate(gene_names) if gene not in genes_to_exclude]
        gene_names = [gene_names[i] for i in filtered_indices]
        predicted_deletions = predicted_deletions[filtered_indices]
        predicted_probs = predicted_probs[filtered_indices]
        true_labels = true_labels[filtered_indices]
    
    return gene_names, predicted_deletions, predicted_probs, true_labels

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
        genes_to_exclude = [gene for gene in gene_count if gene_count[gene]['0'] / gene_total[gene] >= bound**3 or gene_count[gene]['1'] / gene_total[gene] >= bound**3]
        
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

def print_predicted_gene_deletions(gene_names, predicted_deletions, predicted_probs, true_labels, metabolite_name):
    # Print predictions
    print(f"Predicted gene deletions for metabolite '{metabolite_name}':")
    num_genes = len(gene_names)
    for gene_name, deletion, prob, true_label in zip(gene_names, predicted_deletions, predicted_probs, true_labels):
        print(f"Gene: {gene_name}, Deletion: {deletion}, Probability: {prob:.4f}")
    
    # Calculate overall accuracy
    num_correct = np.sum(predicted_deletions == true_labels)
    accuracy = (num_correct / num_genes) * 100 if num_genes > 0 else 0.0
    
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

def plot_gene_deletions_heatmap(gene_names, predicted_deletions, true_labels, metabolite_name):
    # Create a matrix where 0 is green and 1 is red for predicted and true labels
    heatmap_data = np.zeros((len(gene_names), 2))
    heatmap_data[:, 0] = predicted_deletions
    heatmap_data[:, 1] = true_labels
    
    # Create a DataFrame for Seaborn heatmap
    df = pd.DataFrame(heatmap_data, index=gene_names, columns=['Predicted Deletion', 'True Label'])
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(df, cmap=['green', 'red'], cbar=True, annot=False)
    
    # Manually adjust color bar ticks and labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['0 (Not Deleted)', '1 (Deleted)'])
    
    plt.title(f'Predicted vs. True Deletions for Metabolite "{metabolite_name}"')
    plt.xlabel('Labels')
    plt.ylabel('Genes')
    plt.yticks(rotation=0)  # Rotate y-axis labels for better readability
    plt.show()
    
def read_gene_necessity(csv_path):
    gene_necessity = {}
    with open(csv_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            gene_necessity[row['Remaining gene']] = row['Necessity']
    return gene_necessity

def find_prediction_errors(gene_names, predicted_deletions, true_labels, gene_necessity):
    errors = []
    mistake_genes = []
    
    for gene_name, predicted, true_label in zip(gene_names, predicted_deletions, true_labels):
        if predicted != true_label:
            necessity_type = gene_necessity.get(gene_name, 'Unknown')
            errors.append((gene_name, predicted, true_label, necessity_type))
            if gene_name not in mistake_genes:
                mistake_genes.append(gene_name)
                
    return errors

def print_prediction_errors(errors):
    if errors:
        print("Prediction Errors:")
        error_types = defaultdict(int)
        total_errors = len(errors)
        
        for gene_name, predicted, true_label, necessity_type in errors:
            if true_label == 0 and predicted == 1:
                error_type = 'False Positive'
            elif true_label == 1 and predicted == 0:
                error_type = 'False Negative'
            else:
                error_type = 'Unknown'
                
            print(f"Gene: {gene_name}, Predicted: {predicted}, True Label: {true_label}, Necessity Type: {necessity_type}, Error Type: {error_type}")
            error_types[necessity_type] += 1
        
        print("\nMistake Ratio Summary:")
        for necessity_type, count in error_types.items():
            ratio = (count / total_errors) * 100
            print(f"Necessity Type {necessity_type}: {count} errors ({ratio:.2f}%)")
    else:
        print("No prediction errors found.")
        
def calculate_metrics_for_val_metabolites(model, val_metabolites, metabolite_names, gene_sequences, smiles_features, relationships, device, genes_to_exclude):
    total_correct = 0
    total_genes = 0
    total_true_positive = [0, 0]  # For classes 0 and 1
    total_false_positive = [0, 0]  # For classes 0 and 1
    total_true_negative = [0, 0]  # For classes 0 and 1
    total_false_negative = [0, 0]  # For classes 0 and 1
    
    metabolite_metrics = []
    
    for metabolite_name in val_metabolites:
        gene_names, predicted_deletions, predicted_probs, true_labels = predict_gene_deletions_for_metabolite(
            model, metabolite_name, metabolite_names, gene_sequences, smiles_features, relationships, device, genes_to_exclude
        )
        
        if len(gene_names) > 0:  # Ensure there are genes to evaluate
            # Calculate Accuracy
            num_correct = np.sum(predicted_deletions == true_labels)  # Correct predictions
            accuracy = (num_correct / len(gene_names)) * 100
            total_correct += num_correct
            total_genes += len(gene_names)

            # Calculate True Positives, False Positives, True Negatives, and False Negatives for each class
            true_positive = [0, 0]
            false_positive = [0, 0]
            true_negative = [0, 0]
            false_negative = [0, 0]
            
            for label in [0, 1]:
                true_positive[label] = np.sum((predicted_deletions == label) & (true_labels == label))
                false_positive[label] = np.sum((predicted_deletions == label) & (true_labels != label))
                true_negative[label] = np.sum((predicted_deletions != label) & (true_labels != label))
                false_negative[label] = np.sum((predicted_deletions != label) & (true_labels == label))
                
                total_true_positive[label] += true_positive[label]
                total_false_positive[label] += false_positive[label]
                total_true_negative[label] += true_negative[label]
                total_false_negative[label] += false_negative[label]
            
            # Calculate Precision, Recall, and F1 Score for each class
            precision = []
            recall = []
            f1_score = []
            
            for label in [0, 1]:
                precision_val = (true_positive[label] / (true_positive[label] + false_positive[label]) * 100) if (true_positive[label] + false_positive[label]) > 0 else 0.0
                recall_val = (true_positive[label] / (true_positive[label] + false_negative[label]) * 100) if (true_positive[label] + false_negative[label]) > 0 else 0.0
                f1_score_val = (2 * precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0
                
                precision.append(precision_val)
                recall.append(recall_val)
                f1_score.append(f1_score_val)
            
            # Macro-Averaged Metrics
            avg_precision = np.mean(precision)
            avg_recall = np.mean(recall)
            avg_f1_score = np.mean(f1_score)
            
            # Store metrics for each metabolite
            metabolite_metrics.append({
                'accuracy': accuracy,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1_score
            })
            
            # Print results for each metabolite
            print(f"Metabolite: {metabolite_name}")
            print(f"  Number of Non-fixed Genes: {len(gene_names)}")
            # Print first 10 labels (true and predicted)
            print(f"  First 10 True Gene Status: {true_labels[:10]}")
            print(f"  First 10 Predicted Gene Status: {predicted_deletions[:10]}")
            print(f"  Correct Predictions: {num_correct}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Precision: {avg_precision:.2f}% (Macro-Averaged)")
            print(f"  Recall: {avg_recall:.2f}% (Macro-Averaged)")
            print(f"  F1 Score: {avg_f1_score:.2f}% (Macro-Averaged)")
            print()

    # Calculate overall metrics across all validation metabolites
    overall_accuracy = (total_correct / total_genes) * 100 if total_genes > 0 else 0.0
    
    # Micro-Averaged Metrics
    total_true_positive_sum = np.sum(total_true_positive)
    total_false_positive_sum = np.sum(total_false_positive)
    total_true_negative_sum = np.sum(total_true_negative)
    total_false_negative_sum = np.sum(total_false_negative)
    
    micro_precision = (total_true_positive_sum / (total_true_positive_sum + total_false_positive_sum) * 100) if (total_true_positive_sum + total_false_positive_sum) > 0 else 0.0
    micro_recall = (total_true_positive_sum / (total_true_positive_sum + total_false_negative_sum) * 100) if (total_true_positive_sum + total_false_negative_sum) > 0 else 0.0
    micro_f1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # Average the metrics across all metabolites (Macro-Averaged)
    average_precision = np.mean([metrics['precision'] for metrics in metabolite_metrics])
    average_recall = np.mean([metrics['recall'] for metrics in metabolite_metrics])
    average_f1_score = np.mean([metrics['f1_score'] for metrics in metabolite_metrics])
    
    # Print overall results
    print()
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Macro-Averaged Precision: {average_precision:.2f}%")
    print(f"Macro-Averaged Recall: {average_recall:.2f}%")
    print(f"Macro-Averaged F1 Score: {average_f1_score:.2f}%")

def predict_and_save_all_results(
    model, val_metabolites, metabolite_names, gene_sequences, 
    smiles_features, relationships, device, genes_to_exclude, output_csv_path
):
    # Create a dictionary to store results for all metabolites
    all_results = {}
    # List to track prediction times
    prediction_times = []

    for metabolite_name in val_metabolites:
        print(f"Processing metabolite: {metabolite_name}")
        start_time = time.time()  # Start timing

        try:
            # Predict gene deletions for the current metabolite
            gene_names, predicted_deletions, _, _ = predict_gene_deletions_for_metabolite(
                model, metabolite_name, metabolite_names, gene_sequences, 
                smiles_features, relationships, device, genes_to_exclude
            )
            
            # Add results to the dictionary
            all_results[metabolite_name] = predicted_deletions
        
        except Exception as e:
            print(f"Error processing metabolite {metabolite_name}: {e}")
            continue  # Skip this metabolite in case of an error

        end_time = time.time()  # End timing
        prediction_times.append(end_time - start_time)

    # Compute average time per metabolite
    if prediction_times:
        avg_time_per_metabolite = sum(prediction_times) / len(prediction_times)
        print(f"Average time per metabolite prediction: {avg_time_per_metabolite:.4f} seconds")

    # Write the results to a single CSV file
    with open(output_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header row
        header = ['Metabolite'] + gene_names  # First column is "Metabolite", followed by gene names
        writer.writerow(header)

        # Write one row per metabolite
        for metabolite_name, deletions in all_results.items():
            writer.writerow([metabolite_name] + deletions.tolist())

    print(f"All results saved to {output_csv_path}")
