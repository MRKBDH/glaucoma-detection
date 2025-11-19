"""
Training script for LSTM model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import os

from data_utils import load_fairclip_data, prepare_data_splits
from text_preprocessing import TextPreprocessor, build_vocabulary
from dataset import GlaucomaDataset
from lstm_model import LSTMClassifier

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Hyperparameters
EMBEDDING_DIM = 300
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50  # Start with 20 for testing
MAX_SEQ_LENGTH = 200

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for batch in tqdm(dataloader, desc="Training"):
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
    races = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            races.extend(batch['race'])
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(targets, predictions)
    
    # Calculate binary predictions
    binary_preds = [1 if p >= 0.5 else 0 for p in predictions]
    
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(targets, binary_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return avg_loss, auc, sensitivity, specificity, predictions, targets, races


def evaluate_by_race(predictions, targets, races, race_name):
    """Evaluate metrics for a specific race"""
    race_preds = [p for p, r in zip(predictions, races) if r == race_name]
    race_targets = [t for t, r in zip(targets, races) if r == race_name]
    
    if len(race_preds) == 0 or len(set(race_targets)) < 2:
        return None, None, None, 0
    
    auc = roc_auc_score(race_targets, race_preds)
    binary_preds = [1 if p >= 0.5 else 0 for p in race_preds]
    
    tn, fp, fn, tp = confusion_matrix(race_targets, binary_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return auc, sensitivity, specificity, len(race_preds)


def main():
    print("="*60)
    print("LSTM MODEL TRAINING FOR GLAUCOMA DETECTION")
    print("="*60)
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    df = load_fairclip_data()
    train_df, val_df, test_df = prepare_data_splits(df)
    
    # Step 2: Preprocess text
    print("\n[2/6] Preprocessing text...")
    preprocessor = TextPreprocessor()
    
    train_texts = [preprocessor.preprocess(text) for text in train_df['note'].values]
    val_texts = [preprocessor.preprocess(text) for text in val_df['note'].values]
    test_texts = [preprocessor.preprocess(text) for text in test_df['note'].values]
    
    print("Sample preprocessed text:")
    print(train_texts[0][:200], "...")
    
    # Step 3: Build vocabulary
    print("\n[3/6] Building vocabulary...")
    word2idx, vocab = build_vocabulary(train_texts, min_freq=2)
    vocab_size = len(vocab)
    
    # Step 4: Create datasets
    print("\n[4/6] Creating datasets...")
    train_dataset = GlaucomaDataset(
        train_texts, 
        train_df['label'].values,
        train_df['race'].values,
        word2idx,
        MAX_SEQ_LENGTH
    )
    
    val_dataset = GlaucomaDataset(
        val_texts,
        val_df['label'].values,
        val_df['race'].values,
        word2idx,
        MAX_SEQ_LENGTH
    )
    
    test_dataset = GlaucomaDataset(
        test_texts,
        test_df['label'].values,
        test_df['race'].values,
        word2idx,
        MAX_SEQ_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Step 5: Create model
    print("\n[5/6] Creating LSTM model...")
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Step 6: Training loop
    print("\n[6/6] Training model...")
    print("="*60)
    
    best_val_auc = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        
        # Train
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_auc, val_sens, val_spec, _, _, _ = evaluate(model, val_loader, criterion, device)
        
        print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        print(f"  Val Sensitivity: {val_sens:.4f}, Val Specificity: {val_spec:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_lstm_model.pth')
            print(f"  ✓ Best model saved! (AUC: {best_val_auc:.4f})")
    
    # Load best model for testing
    print("\n" + "="*60)
    print("TESTING ON TEST SET")
    print("="*60)
    
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    test_loss, test_auc, test_sens, test_spec, predictions, targets, races = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nOverall Test Performance:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Sensitivity: {test_sens:.4f}")
    print(f"  Specificity: {test_spec:.4f}")
    
    # Evaluate by race
    print(f"\nPerformance by Race:")
    for race_name in ['white', 'black', 'asian']:
        auc, sens, spec, count = evaluate_by_race(predictions, targets, races, race_name)
        if auc is not None:
            print(f"  {race_name.capitalize()}:")
            print(f"    AUC: {auc:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}, N={count}")
    
    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
