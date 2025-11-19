"""
Visualization script for LSTM results
Creates ROC curves, training history, and comparison plots
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from data_utils import load_fairclip_data, prepare_data_splits
from text_preprocessing import TextPreprocessor, build_vocabulary
from dataset import GlaucomaDataset
from lstm_model import LSTMClassifier

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Model parameters (must match trained model)
EMBEDDING_DIM = 300
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


def get_predictions(model, dataloader, device):
    """Get predictions and labels from dataloader"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_races = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting predictions"):
            inputs = batch['input_ids'].to(device)
            labels = batch['label']
            
            outputs = model(inputs)
            
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_races.extend(batch['race'])
    
    return np.array(all_predictions), np.array(all_labels), all_races


def plot_roc_curve(y_true, y_pred, title="ROC Curve", save_path="roc_curve.png"):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_roc_by_race(y_true, y_pred, races, save_path="roc_by_race.png"):
    """Plot ROC curves for each race"""
    plt.figure(figsize=(10, 6))
    
    colors = {'white': '#1f77b4', 'black': '#ff7f0e', 'asian': '#2ca02c'}
    
    for race_name, color in colors.items():
        # Filter by race
        race_mask = [r == race_name for r in races]
        if sum(race_mask) == 0:
            continue
            
        race_true = y_true[race_mask]
        race_pred = y_pred[race_mask]
        
        if len(set(race_true)) < 2:
            continue
        
        fpr, tpr, _ = roc_curve(race_true, race_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{race_name.capitalize()} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves by Demographic Group', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Plot confusion matrix"""
    binary_pred = (y_pred >= 0.5).astype(int)
    cm = confusion_matrix(y_true, binary_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Glaucoma', 'Glaucoma'],
                yticklabels=['No Glaucoma', 'Glaucoma'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - LSTM Model', fontsize=14, fontweight='bold')
    
    # Add accuracy
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.text(1, -0.3, f'Accuracy: {accuracy:.4f}', 
             ha='center', fontsize=11, transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_performance_by_race(predictions, labels, races, save_path="performance_by_race.png"):
    """Plot AUC, Sensitivity, Specificity by race"""
    from sklearn.metrics import roc_auc_score
    
    results = {}
    
    for race_name in ['white', 'black', 'asian']:
        race_mask = [r == race_name for r in races]
        if sum(race_mask) == 0:
            continue
            
        race_pred = predictions[race_mask]
        race_true = labels[race_mask]
        
        if len(set(race_true)) < 2:
            continue
        
        # Calculate metrics
        auc_score = roc_auc_score(race_true, race_pred)
        binary_pred = (race_pred >= 0.5).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(race_true, binary_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results[race_name] = {
            'AUC': auc_score,
            'Sensitivity': sensitivity,
            'Specificity': specificity
        }
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    races_list = list(results.keys())
    metrics = ['AUC', 'Sensitivity', 'Specificity']
    
    x = np.arange(len(races_list))
    width = 0.25
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, metric in enumerate(metrics):
        values = [results[race][metric] for race in races_list]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric, 
                      color=colors[i], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Demographic Group', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics by Demographic Group', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in races_list])
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_results_table(predictions, labels, races, save_path="results_table.txt"):
    """Create formatted results table"""
    from sklearn.metrics import roc_auc_score
    
    # Overall metrics
    overall_auc = roc_auc_score(labels, predictions)
    binary_pred = (predictions >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, binary_pred).ravel()
    overall_sens = tp / (tp + fn)
    overall_spec = tn / (tn + fp)
    
    # By race
    race_results = {}
    for race_name in ['white', 'black', 'asian']:
        race_mask = [r == race_name for r in races]
        if sum(race_mask) == 0:
            continue
            
        race_pred = predictions[race_mask]
        race_true = labels[race_mask]
        
        if len(set(race_true)) < 2:
            continue
        
        auc_score = roc_auc_score(race_true, race_pred)
        binary_pred = (race_pred >= 0.5).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(race_true, binary_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        race_results[race_name] = (auc_score, sensitivity, specificity, sum(race_mask))
    
    # Write to file
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("LSTM MODEL RESULTS - GLAUCOMA DETECTION\n")
        f.write("="*70 + "\n\n")
        
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"AUC:         {overall_auc:.4f}\n")
        f.write(f"Sensitivity: {overall_sens:.4f}\n")
        f.write(f"Specificity: {overall_spec:.4f}\n")
        f.write(f"Total samples: {len(labels)}\n\n")
        
        f.write("PERFORMANCE BY DEMOGRAPHIC GROUP\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Group':<12} {'AUC':<10} {'Sensitivity':<15} {'Specificity':<15} {'N':<8}\n")
        f.write("-"*70 + "\n")
        
        for race_name, (auc_val, sens, spec, count) in race_results.items():
            f.write(f"{race_name.capitalize():<12} {auc_val:<10.4f} {sens:<15.4f} {spec:<15.4f} {count:<8}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Saved: {save_path}")


def main():
    print("="*60)
    print("CREATING VISUALIZATIONS FOR LSTM MODEL")
    print("="*60)
    
    # Create output directory
    os.makedirs("figures", exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading data and model...")
    df = load_fairclip_data()
    train_df, val_df, test_df = prepare_data_splits(df)
    
    # Preprocess
    preprocessor = TextPreprocessor()
    train_texts = [preprocessor.preprocess(text) for text in train_df['note'].values]
    test_texts = [preprocessor.preprocess(text) for text in test_df['note'].values]
    
    # Build vocabulary
    word2idx, vocab = build_vocabulary(train_texts, min_freq=2)
    vocab_size = len(vocab)
    
    # Create test dataset
    test_dataset = GlaucomaDataset(
        test_texts,
        test_df['label'].values,
        test_df['race'].values,
        word2idx,
        MAX_SEQ_LENGTH
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    model.load_state_dict(torch.load('best_lstm_256.pth'))
    print("✓ Model loaded")
    
    # Get predictions
    print("\n[2/4] Getting predictions...")
    predictions, labels, races = get_predictions(model, test_loader, device)
    
    # Create visualizations
    print("\n[3/4] Creating visualizations...")
    
    plot_roc_curve(labels, predictions, 
                   title="ROC Curve - LSTM Model",
                   save_path="figures/roc_curve.png")
    
    plot_roc_by_race(labels, predictions, races,
                     save_path="figures/roc_by_race.png")
    
    plot_confusion_matrix(labels, predictions,
                         save_path="figures/confusion_matrix.png")
    
    plot_performance_by_race(predictions, labels, races,
                            save_path="figures/performance_by_race.png")
    
    # Create results table
    print("\n[4/4] Creating results table...")
    create_results_table(predictions, labels, races,
                        save_path="figures/results_table.txt")
    
    print("\n" + "="*60)
    print("✓ All visualizations created successfully!")
    print("="*60)
    print("\nGenerated files in 'figures/' directory:")
    print("  - roc_curve.png")
    print("  - roc_by_race.png")
    print("  - confusion_matrix.png")
    print("  - performance_by_race.png")
    print("  - results_table.txt")
    print("\nYou can use these figures in your report!")


if __name__ == "__main__":
    main()
