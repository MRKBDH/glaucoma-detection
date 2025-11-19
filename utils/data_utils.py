"""
Data loading and preprocessing utilities for FairCLIP dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_fairclip_data(data_path='data/clinical_notes.csv'):
    """
    Load the FairCLIP clinical notes dataset
    
    Returns:
    --------
    DataFrame with the clinical notes data
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(df)}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    return df


def explore_dataset(df):
    """
    Print dataset statistics
    """
    print("\n" + "="*60)
    print("DATASET EXPLORATION")
    print("="*60)
    
    # Basic info
    print(f"\nTotal samples: {len(df)}")
    print(f"\nColumn names: {df.columns.tolist()}")
    
    # Check for glaucoma labels
    if 'glaucoma' in df.columns:
        print(f"\nGlaucoma distribution:")
        print(df['glaucoma'].value_counts())
        glaucoma_pct = (df['glaucoma'] == 'yes').sum() / len(df) * 100
        print(f"Glaucoma positive: {glaucoma_pct:.1f}%")
    
    # Check for race distribution
    if 'race' in df.columns:
        print(f"\nRace distribution:")
        print(df['race'].value_counts())
    
    # Check for train/test split
    if 'use' in df.columns:
        print(f"\nData split:")
        print(df['use'].value_counts())
    
    # Check missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Sample data
    print(f"\nFirst row sample:")
    print(df.iloc[0][['race', 'glaucoma', 'use']])
    
    print("\n" + "="*60)


def prepare_data_splits(df):
    """
    Prepare train/validation/test splits based on 'use' column
    """
    print("\nPreparing data splits...")
    
    # Convert glaucoma to binary (yes=1, no=0)
    df['label'] = (df['glaucoma'] == 'yes').astype(int)
    
    # Split based on 'use' column (already split in dataset)
    train_df = df[df['use'] == 'training'].copy()
    val_df = df[df['use'] == 'validation'].copy()
    test_df = df[df['use'] == 'test'].copy()
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Show distribution
    print(f"\nTraining - Glaucoma: {train_df['label'].sum()} positive, {len(train_df)-train_df['label'].sum()} negative")
    print(f"Validation - Glaucoma: {val_df['label'].sum()} positive, {len(val_df)-val_df['label'].sum()} negative")
    print(f"Test - Glaucoma: {test_df['label'].sum()} positive, {len(test_df)-test_df['label'].sum()} negative")
    
    # Show race distribution per split
    print(f"\nRace distribution in training:")
    print(train_df['race'].value_counts())
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Load data
    df = load_fairclip_data()
    
    # Explore dataset
    explore_dataset(df)
    
    # Prepare splits
    train_df, val_df, test_df = prepare_data_splits(df)
    
    print("\n✓ Data loading successful!")
