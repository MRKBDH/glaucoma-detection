"""
PyTorch Dataset for clinical notes
"""

import torch
from torch.utils.data import Dataset


class GlaucomaDataset(Dataset):
    """PyTorch Dataset for glaucoma detection"""
    
    def __init__(self, texts, labels, races, word2idx, max_length=200):
        """
        Args:
            texts: List of preprocessed text strings
            labels: List of binary labels (0 or 1)
            races: List of race labels for fairness evaluation
            word2idx: Dictionary mapping words to indices
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.races = races
        self.word2idx = word2idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        race = self.races[idx]
        
        # Convert text to indices
        indices = []
        for word in text.split()[:self.max_length]:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx['<UNK>'])  # Unknown word
        
        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices += [self.word2idx['<PAD>']] * (self.max_length - len(indices))
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float),
            'race': race
        }


if __name__ == "__main__":
    # Test the dataset
    print("Testing GlaucomaDataset...")
    
    # Sample data
    texts = [
        "patient elevated pressure glaucoma",
        "normal examination healthy optic nerve",
        "increased cup disc ratio"
    ]
    labels = [1, 0, 1]
    races = ['white', 'black', 'asian']
    
    # Create simple vocabulary
    word2idx = {'<PAD>': 0, '<UNK>': 1, 'patient': 2, 'elevated': 3, 
                'pressure': 4, 'glaucoma': 5, 'normal': 6, 'examination': 7,
                'healthy': 8, 'optic': 9, 'nerve': 10, 'increased': 11,
                'cup': 12, 'disc': 13, 'ratio': 14}
    
    # Create dataset
    dataset = GlaucomaDataset(texts, labels, races, word2idx, max_length=10)
    
    # Test getting an item
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Input IDs: {sample['input_ids']}")
    print(f"  Label: {sample['label']}")
    print(f"  Race: {sample['race']}")
    
    print(f"\nDataset length: {len(dataset)}")
    print("✓ Dataset works!")
