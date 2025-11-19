"""
Experiment with different LSTM hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import json

from data_utils import load_fairclip_data, prepare_data_splits
from text_preprocessing import TextPreprocessor, build_vocabulary
from dataset import GlaucomaDataset
from lstm_model import LSTMClassifier

# Set random seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Fixed parameters
EMBEDDING_DIM = 300
BATCH_SIZE = 32
NUM_EPOCHS = 30
MAX_SEQ_LENGTH = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(outputs.detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(targets, predictions)
    
    return avg_loss, auc


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(targets, predictions)
    
    binary_preds = [1 if p >= 0.5 else 0 for p in predictions]
    tn, fp, fn, tp = confusion_matrix(targets, binary_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return avg_loss, auc, sensitivity, specificity


def train_with_config(train_loader, val_loader, vocab_size, config, config_name):
    """Train model with specific configuration"""
    print(f"\n{'='*60}")
    print(f"Testing Configuration: {config_name}")
    print(f"{'='*60}")
    print(f"Config: {config}")
    
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_sens, val_spec = evaluate(model, val_loader, criterion, device)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Val AUC: {val_auc:.4f}, "
                  f"Sens: {val_sens:.4f}, Spec: {val_spec:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_metrics = {
                'val_auc': val_auc,
                'val_sensitivity': val_sens,
                'val_specificity': val_spec,
                'epoch': epoch + 1
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest validation AUC: {best_val_auc:.4f} at epoch {best_metrics['epoch']}")
    
    return best_metrics


def main():
    print("="*60)
    print("LSTM HYPERPARAMETER EXPERIMENTS")
    print("="*60)
    
    # Load and prepare data
    print("\nLoading and preprocessing data...")
    df = load_fairclip_data()
    train_df, val_df, test_df = prepare_data_splits(df)
    
    preprocessor = TextPreprocessor()
    train_texts = [preprocessor.preprocess(text) for text in train_df['note'].values]
    val_texts = [preprocessor.preprocess(text) for text in val_df['note'].values]
    
    word2idx, vocab = build_vocabulary(train_texts, min_freq=2)
    vocab_size = len(vocab)
    
    train_dataset = GlaucomaDataset(train_texts, train_df['label'].values, 
                                    train_df['race'].values, word2idx, MAX_SEQ_LENGTH)
    val_dataset = GlaucomaDataset(val_texts, val_df['label'].values, 
                                  val_df['race'].values, word2idx, MAX_SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Define configurations to test
    configs = {
        'Baseline (Original)': {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 0.001
        },
        'Lower LR': {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 0.0005
        },
        'Higher Dropout': {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.4,
            'learning_rate': 0.001
        },
        'Larger Hidden': {
            'hidden_size': 256,
            'num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 0.001
        },
        'More Layers': {
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.3,
            'learning_rate': 0.001
        },
        'Smaller Model': {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001
        }
    }
    
    # Test all configurations
    results = {}
    
    for config_name, config in configs.items():
        metrics = train_with_config(train_loader, val_loader, vocab_size, config, config_name)
        results[config_name] = {
            'config': config,
            'metrics': metrics
        }
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n{'Configuration':<20} {'Val AUC':<10} {'Sensitivity':<12} {'Specificity':<12} {'Best Epoch':<12}")
    print("-" * 70)
    
    for config_name, result in results.items():
        metrics = result['metrics']
        print(f"{config_name:<20} {metrics['val_auc']:<10.4f} "
              f"{metrics['val_sensitivity']:<12.4f} {metrics['val_specificity']:<12.4f} "
              f"{metrics['epoch']:<12}")
    
    # Find best configuration
    best_config = max(results.items(), key=lambda x: x[1]['metrics']['val_auc'])
    print("\n" + "="*60)
    print(f"🏆 BEST CONFIGURATION: {best_config[0]}")
    print(f"   Val AUC: {best_config[1]['metrics']['val_auc']:.4f}")
    print("="*60)
    
    # Save results
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Results saved to experiment_results.json")


if __name__ == "__main__":
    main()
