import torch
import torch.nn as nn 

class CombinedModel(nn.Module):
    def __init__(self, gene_lstm, smiles_model, hidden_dim):
        super(CombinedModel, self).__init__()
        self.gene_lstm = gene_lstm
        self.smiles_model = smiles_model
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, gene_seq, smiles_feat):
        # Forward pass for GeneLSTM
        if any(p.requires_grad for p in self.gene_lstm.parameters()):
            gene_feat, reconstructed_gene = self.gene_lstm(gene_seq)
        else:
            gene_feat, reconstructed_gene = None, None
        
        # Forward pass for SMILESLSTM
        if any(p.requires_grad for p in self.smiles_model.parameters()):
            smiles_feat, reconstructed_smiles = self.smiles_model(smiles_feat)
        else:
            smiles_feat, reconstructed_smiles = None, None
        
        # Combine features
        if gene_feat is not None and smiles_feat is not None:
            combined_feat = gene_feat * smiles_feat
        elif gene_feat is not None:
            combined_feat = gene_feat
        elif smiles_feat is not None:
            combined_feat = smiles_feat
        else:
            raise ValueError("Both models cannot be frozen at the same time.")
        
        # Final prediction with sigmoid activation for binary classification
        output = torch.sigmoid(self.fc(combined_feat))
        
        # Return final output and reconstruction losses
        return output, reconstructed_gene, reconstructed_smiles