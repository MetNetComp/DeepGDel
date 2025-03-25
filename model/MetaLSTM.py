import torch
import torch.nn as nn  

# Metabolic features LSTM model
class MetaLSTM(nn.Module):
    def __init__(self, vocab_size, vocab_embedding_dim, hidden_dim):
        super(MetaLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_embedding_dim)
        self.lstm = nn.LSTM(vocab_embedding_dim, hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
        # Reverse LSTM Decoder
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_dim, vocab_size)  # Same vocab size as input

    def forward(self, x):
        # Embedding the input
        embedded = self.embedding(x.long())  # Ensure input is of type long for embedding layer
        
        # LSTM forward pass
        lstm_out, (hidden, _) = self.lstm(embedded)  # lstm_out: (batch_size, seq_length, hidden_dim)
        
        # Layer normalization on the LSTM output
        lstm_out = self.layer_norm(lstm_out)  # Apply LayerNorm over the output of each time step
        
        # Pooling the LSTM output (mean over time steps)
        pooled_output = lstm_out.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Fully connected layer to get the final output (for classification task)
        output = self.fc(pooled_output)  # (batch_size, hidden_dim)
        
        # Decoder LSTM to reconstruct the input sequence
        decoder_out, _ = self.decoder_lstm(lstm_out)  # (batch_size, seq_length, hidden_dim)
        
        # Project to vocab size for each time step in the sequence
        reconstructed_input = self.decoder_fc(decoder_out)  # (batch_size, seq_length, vocab_size)
        
        # Apply argmax along the vocabulary dimension to get predicted class indices
        reconstructed_input_indices = reconstructed_input.argmax(dim=2)  # (batch_size, seq_length)
        
        return output, reconstructed_input_indices