"""
1D CNN Model for Glaucoma Detection
"""

import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    """1D CNN model for glaucoma classification"""
    
    def __init__(self, vocab_size, embedding_dim=300, num_filters=128, 
                 kernel_sizes=[3, 4, 5], dropout=0.3):
        super(CNN1DClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.fc1 = nn.Linear(len(kernel_sizes) * num_filters, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        # Permute for Conv1d: (batch_size, embedding_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolution and max pooling for each kernel size
        conv_results = []
        for conv in self.convs:
            conv_out = self.relu(conv(embedded))
            # conv_out shape: (batch_size, num_filters, seq_len - kernel_size + 1)
            
            pooled = torch.max(conv_out, dim=2)[0]
            # pooled shape: (batch_size, num_filters)
            
            conv_results.append(pooled)
        
        # Concatenate all pooled results
        out = torch.cat(conv_results, dim=1)
        # out shape: (batch_size, len(kernel_sizes) * num_filters)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out.squeeze()


if __name__ == "__main__":
    # Test the model
    print("Testing 1D CNN Model...")
    
    # Model parameters
    vocab_size = 10000
    batch_size = 4
    seq_length = 200
    
    # Create model
    model = CNN1DClassifier(vocab_size=vocab_size)
    
    # Create dummy input
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    print(f"\n✓ 1D CNN model works!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
