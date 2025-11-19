"""
GRU Model for Glaucoma Detection
"""

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """GRU model for glaucoma classification"""
    
    def __init__(self, vocab_size, embedding_dim=300, hidden_size=128, 
                 num_layers=2, dropout=0.3):
        super(GRUClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=True
        )
        
        # *2 because bidirectional
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        gru_out, hidden = self.gru(embedded)
        # hidden shape: (num_layers*2, batch_size, hidden_size)
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        # hidden shape: (batch_size, hidden_size*2)
        
        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out.squeeze()


if __name__ == "__main__":
    # Test the model
    print("Testing GRU Model...")
    
    # Model parameters
    vocab_size = 10000
    batch_size = 4
    seq_length = 200
    
    # Create model
    model = GRUClassifier(vocab_size=vocab_size)
    
    # Create dummy input
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    print(f"\n✓ GRU model works!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
