"""
Create professional architecture diagrams for LSTM, GRU, and CNN models
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


def draw_box(ax, x, y, width, height, text, color, text_color='white'):
    """Draw a colored box with text"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1", 
        facecolor=color, 
        edgecolor='black', 
        linewidth=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', 
            fontsize=10, fontweight='bold', color=text_color)


def draw_arrow(ax, x1, y1, x2, y2, style='->', color='black', linewidth=2):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=color,
        linewidth=linewidth,
        mutation_scale=20
    )
    ax.add_patch(arrow)


def create_lstm_architecture():
    """Create LSTM architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'LSTM Architecture for Glaucoma Detection', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Input Layer
    y_pos = 10
    draw_box(ax, 5, y_pos, 3, 0.8, 'Input: Clinical Notes\n(Text Sequence)', '#3498db', 'white')
    ax.text(8.5, y_pos, 'Shape: [batch, seq_len]', ha='left', fontsize=9, style='italic')
    
    # Arrow
    draw_arrow(ax, 5, y_pos-0.5, 5, y_pos-1.3)
    
    # Text Preprocessing
    y_pos = 8.5
    draw_box(ax, 5, y_pos, 3, 0.8, 'Text Preprocessing\n(Tokenization, Cleaning)', '#95a5a6', 'white')
    ax.text(8.5, y_pos, 'Lowercase, remove\nspecial chars', ha='left', fontsize=8, style='italic')
    
    # Arrow
    draw_arrow(ax, 5, y_pos-0.5, 5, y_pos-1.3)
    
    # Embedding Layer
    y_pos = 7
    draw_box(ax, 5, y_pos, 3, 0.8, 'Embedding Layer\n(300-dimensional)', '#9b59b6', 'white')
    ax.text(8.5, y_pos, 'Word → Vector\nVocab size: ~10k', ha='left', fontsize=8, style='italic')
    
    # Arrow
    draw_arrow(ax, 5, y_pos-0.5, 5, y_pos-1.3)
    
    # LSTM Layer 1
    y_pos = 5.5
    draw_box(ax, 5, y_pos, 3.5, 1.2, 'LSTM Layer 1\nHidden Size: 256\nBidirectional', '#e74c3c', 'white')
    ax.text(8.8, y_pos+0.2, '512 units', ha='left', fontsize=8, style='italic')
    ax.text(8.8, y_pos-0.2, '(256 fwd + 256 bwd)', ha='left', fontsize=8, style='italic')
    
    # Small LSTM cell diagram
    cell_x, cell_y = 2, 5.5
    ax.add_patch(plt.Rectangle((cell_x-0.3, cell_y-0.4), 0.6, 0.8, 
                               facecolor='#c0392b', edgecolor='black', linewidth=1.5))
    ax.text(cell_x, cell_y+0.15, 'h', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    ax.text(cell_x, cell_y-0.15, 'c', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    ax.text(cell_x-0.8, cell_y, 'Cell\nState', ha='center', fontsize=7, style='italic')
    
    # Arrow
    draw_arrow(ax, 5, y_pos-0.7, 5, y_pos-1.5)
    
    # Dropout 1
    y_pos = 3.8
    draw_box(ax, 5, y_pos, 2, 0.6, 'Dropout (0.3)', '#7f8c8d', 'white')
    
    # Arrow
    draw_arrow(ax, 5, y_pos-0.4, 5, y_pos-1.0)
    
    # LSTM Layer 2
    y_pos = 2.5
    draw_box(ax, 5, y_pos, 3.5, 1.2, 'LSTM Layer 2\nHidden Size: 256\nBidirectional', '#e74c3c', 'white')
    
    # Arrow
    draw_arrow(ax, 5, y_pos-0.7, 5, y_pos-1.5)
    
    # Dropout 2
    y_pos = 0.6
    draw_box(ax, 5, y_pos, 2, 0.6, 'Dropout (0.3)', '#7f8c8d', 'white')
    
    # Arrows to FC
    draw_arrow(ax, 5, 0.2, 3.5, -0.5)
    
    # Fully Connected Layers (side by side)
    y_pos = -1
    draw_box(ax, 2.5, y_pos, 2, 0.8, 'FC Layer\n(512 → 128)', '#16a085', 'white')
    draw_arrow(ax, 3.5, y_pos, 4.5, y_pos)
    draw_box(ax, 5.5, y_pos, 2, 0.8, 'ReLU', '#27ae60', 'white')
    draw_arrow(ax, 6.5, y_pos, 7.5, y_pos)
    
    # Dropout 3
    draw_box(ax, 8.5, y_pos, 1.5, 0.8, 'Dropout\n(0.3)', '#7f8c8d', 'white')
    
    # Arrow down
    draw_arrow(ax, 5, y_pos-0.5, 5, y_pos-1.3)
    
    # Output Layer
    y_pos = -2.5
    draw_box(ax, 5, y_pos, 2.5, 0.8, 'Output Layer (Sigmoid)\n1 unit', '#2ecc71', 'white')
    ax.text(8, y_pos, 'P(Glaucoma)', ha='left', fontsize=9, style='italic')
    
    # Model info box
    info_text = (
        'Model Parameters:\n'
        '• Embedding: 300D\n'
        '• Hidden Size: 256 (×2 for bidirectional)\n'
        '• Layers: 2\n'
        '• Total Params: ~5.62M'
    )
    ax.text(0.5, -1, info_text, fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('architecture_lstm.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: architecture_lstm.png")
    plt.close()


def create_gru_architecture():
    """Create GRU architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'GRU Architecture for Glaucoma Detection', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Input Layer
    y_pos = 10
    draw_box(ax, 5, y_pos, 3, 0.8, 'Input: Clinical Notes\n(Text Sequence)', '#3498db', 'white')
    ax.text(8.5, y_pos, 'Shape: [batch, seq_len]', ha='left', fontsize=9, style='italic')
    
    # Arrow
    draw_arrow(ax, 5, y_pos-0.5, 5, y_pos-1.3)
    
    # Text Preprocessing
    y_pos = 8.5
    draw_box(ax, 5, y_pos, 3, 0.8, 'Text Preprocessing\n(Tokenization, Cleaning)', '#95a5a6', 'white')
    ax.text(8.5, y_pos, 'Lowercase, remove\nspecial chars', ha='left', fontsize=8, style='italic')
    
    # Arrow
    draw_arrow(ax, 5, y_pos-0.5, 5, y_pos-1.3)
    
    # Embedding Layer
    y_pos = 7
    draw_box(ax, 5, y_pos, 3, 0.8, 'Embedding Layer\n(300-dimensional)', '#9b59b6', 'white')
    ax.text(8.5, y_pos, 'Word → Vector\nVocab size: ~10k', ha='left', fontsize=8, style='italic')
    
    # Arrow
    draw_arrow(ax, 5, y_pos-0.5, 5, y_pos-1.3)
    
    # GRU Layer 1
    y_pos = 5.5
    draw_box(ax, 5, y_pos, 3.5, 1.2, 'GRU Layer 1\nHidden Size: 256\nBidirectional', '#e67e22', 'white')
    ax.text(8.8, y_pos+0.2, '512 units', ha='left', fontsize=8, style='italic')
    ax.text(8.8, y_pos-0.2, '(256 fwd + 256 bwd)', ha='left', fontsize=8, style='italic')
    
    # Small GRU cell diagram
    cell_x, cell_y = 2, 5.5
    ax.add_patch(plt.Rectangle((cell_x-0.3, cell_y-0.4), 0.6, 0.8, 
                               facecolor='#d35400', edgecolor='black', linewidth=1.5))
    ax.text(cell_x, cell_y, 'h', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    ax.text(cell_x-0.8, cell_y, 'Hidden\nState', ha='center', fontsize=7, style='italic')
    ax.text(cell_x, cell_y-0.7, 'Simpler than LSTM', ha='center', fontsize=7, style='italic')
    
    # Arrow
    draw_arrow(ax, 5, y_pos-0.7, 5, y_pos-1.5)
    
    # Dropout 1
    y_pos = 3.8
    draw_box(ax, 5, y_pos, 2, 0.6, 'Dropout (0.3)', '#7f8c8d', 'white')
    
    # Arrow
    draw_arrow(ax, 5, y_pos-0.4, 5, y_pos-1.0)
    
    # GRU Layer 2
    y_pos = 2.5
    draw_box(ax, 5, y_pos, 3.5, 1.2, 'GRU Layer 2\nHidden Size: 256\nBidirectional', '#e67e22', 'white')
    
    # Arrow
    draw_arrow(ax, 5, y_pos-0.7, 5, y_pos-1.5)
    
    # Dropout 2
    y_pos = 0.6
    draw_box(ax, 5, y_pos, 2, 0.6, 'Dropout (0.3)', '#7f8c8d', 'white')
    
    # Arrows to FC
    draw_arrow(ax, 5, 0.2, 3.5, -0.5)
    
    # Fully Connected Layers (side by side)
    y_pos = -1
    draw_box(ax, 2.5, y_pos, 2, 0.8, 'FC Layer\n(512 → 128)', '#16a085', 'white')
    draw_arrow(ax, 3.5, y_pos, 4.5, y_pos)
    draw_box(ax, 5.5, y_pos, 2, 0.8, 'ReLU', '#27ae60', 'white')
    draw_arrow(ax, 6.5, y_pos, 7.5, y_pos)
    
    # Dropout 3
    draw_box(ax, 8.5, y_pos, 1.5, 0.8, 'Dropout\n(0.3)', '#7f8c8d', 'white')
    
    # Arrow down
    draw_arrow(ax, 5, y_pos-0.5, 5, y_pos-1.3)
    
    # Output Layer
    y_pos = -2.5
    draw_box(ax, 5, y_pos, 2.5, 0.8, 'Output Layer (Sigmoid)\n1 unit', '#2ecc71', 'white')
    ax.text(8, y_pos, 'P(Glaucoma)', ha='left', fontsize=9, style='italic')
    
    # Model info box
    info_text = (
        'Model Parameters:\n'
        '• Embedding: 300D\n'
        '• Hidden Size: 256 (×2 for bidirectional)\n'
        '• Layers: 2\n'
        '• Total Params: ~4.94M\n'
        '• Faster than LSTM (fewer gates)'
    )
    ax.text(0.5, -1, info_text, fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('architecture_gru.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: architecture_gru.png")
    plt.close()


def create_cnn_architecture():
    """Create 1D CNN architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(6, 11.5, '1D CNN Architecture for Glaucoma Detection', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Input Layer
    y_pos = 10
    draw_box(ax, 6, y_pos, 3, 0.8, 'Input: Clinical Notes\n(Text Sequence)', '#3498db', 'white')
    ax.text(9.5, y_pos, 'Shape: [batch, seq_len]', ha='left', fontsize=9, style='italic')
    
    # Arrow
    draw_arrow(ax, 6, y_pos-0.5, 6, y_pos-1.3)
    
    # Text Preprocessing
    y_pos = 8.5
    draw_box(ax, 6, y_pos, 3, 0.8, 'Text Preprocessing\n(Tokenization, Cleaning)', '#95a5a6', 'white')
    ax.text(9.5, y_pos, 'Lowercase, remove\nspecial chars', ha='left', fontsize=8, style='italic')
    
    # Arrow
    draw_arrow(ax, 6, y_pos-0.5, 6, y_pos-1.3)
    
    # Embedding Layer
    y_pos = 7
    draw_box(ax, 6, y_pos, 3, 0.8, 'Embedding Layer\n(300-dimensional)', '#9b59b6', 'white')
    ax.text(9.5, y_pos, 'Word → Vector\nVocab size: ~10k', ha='left', fontsize=8, style='italic')
    
    # Arrow splits into 3
    draw_arrow(ax, 6, y_pos-0.5, 3, y_pos-1.5)
    draw_arrow(ax, 6, y_pos-0.5, 6, y_pos-1.5)
    draw_arrow(ax, 6, y_pos-0.5, 9, y_pos-1.5)
    
    # Parallel Convolution Layers
    y_pos = 5.2
    
    # Conv layer 1 (kernel size 3)
    draw_box(ax, 3, y_pos, 2.2, 1.2, 'Conv1D\nKernel: 3\nFilters: 128', '#f39c12', 'white')
    ax.text(3, y_pos-0.9, '↓', ha='center', fontsize=20)
    draw_box(ax, 3, y_pos-1.5, 2, 0.6, 'ReLU', '#27ae60', 'white')
    ax.text(3, y_pos-2.3, '↓', ha='center', fontsize=20)
    draw_box(ax, 3, y_pos-3, 2, 0.6, 'MaxPool1D', '#3498db', 'white')
    
    # Conv layer 2 (kernel size 4)
    draw_box(ax, 6, y_pos, 2.2, 1.2, 'Conv1D\nKernel: 4\nFilters: 128', '#f39c12', 'white')
    ax.text(6, y_pos-0.9, '↓', ha='center', fontsize=20)
    draw_box(ax, 6, y_pos-1.5, 2, 0.6, 'ReLU', '#27ae60', 'white')
    ax.text(6, y_pos-2.3, '↓', ha='center', fontsize=20)
    draw_box(ax, 6, y_pos-3, 2, 0.6, 'MaxPool1D', '#3498db', 'white')
    
    # Conv layer 3 (kernel size 5)
    draw_box(ax, 9, y_pos, 2.2, 1.2, 'Conv1D\nKernel: 5\nFilters: 128', '#f39c12', 'white')
    ax.text(9, y_pos-0.9, '↓', ha='center', fontsize=20)
    draw_box(ax, 9, y_pos-1.5, 2, 0.6, 'ReLU', '#27ae60', 'white')
    ax.text(9, y_pos-2.3, '↓', ha='center', fontsize=20)
    draw_box(ax, 9, y_pos-3, 2, 0.6, 'MaxPool1D', '#3498db', 'white')
    
    # Visual representation of filters
    for i, x_pos in enumerate([3, 6, 9]):
        kernel_size = [3, 4, 5][i]
        for j in range(3):
            rect_y = y_pos + 0.3 - j*0.15
            ax.add_patch(plt.Rectangle((x_pos-0.4+j*0.15, rect_y), 0.5, 0.08, 
                                      facecolor='yellow', edgecolor='black', linewidth=0.5, alpha=0.6))
    
    # Arrows converge to concatenate
    y_pos = 1.5
    draw_arrow(ax, 3, 2.2, 6, y_pos+0.4)
    draw_arrow(ax, 6, 2.2, 6, y_pos+0.4)
    draw_arrow(ax, 9, 2.2, 6, y_pos+0.4)
    
    # Concatenate
    draw_box(ax, 6, y_pos, 2.5, 0.6, 'Concatenate', '#8e44ad', 'white')
    ax.text(9, y_pos, '384 features\n(128×3)', ha='left', fontsize=8, style='italic')
    
    # Arrow
    draw_arrow(ax, 6, y_pos-0.4, 6, y_pos-1.0)
    
    # Dropout
    y_pos = 0.3
    draw_box(ax, 6, y_pos, 2, 0.6, 'Dropout (0.3)', '#7f8c8d', 'white')
    
    # Arrow
    draw_arrow(ax, 6, y_pos-0.4, 6, y_pos-1.0)
    
    # Fully Connected Layer
    y_pos = -0.9
    draw_box(ax, 6, y_pos, 2.5, 0.8, 'FC Layer (384 → 128)', '#16a085', 'white')
    
    # Arrow
    draw_arrow(ax, 6, y_pos-0.5, 6, y_pos-1.1)
    
    # ReLU
    y_pos = -2.2
    draw_box(ax, 6, y_pos, 2, 0.6, 'ReLU + Dropout', '#27ae60', 'white')
    
    # Arrow
    draw_arrow(ax, 6, y_pos-0.4, 6, y_pos-1.0)
    
    # Output Layer
    y_pos = -3.4
    draw_box(ax, 6, y_pos, 2.5, 0.8, 'Output Layer (Sigmoid)\n1 unit', '#2ecc71', 'white')
    ax.text(9, y_pos, 'P(Glaucoma)', ha='left', fontsize=9, style='italic')
    
    # Model info box
    info_text = (
        'Model Parameters:\n'
        '• Embedding: 300D\n'
        '• Parallel filters: 3, 4, 5\n'
        '• Filters per size: 128\n'
        '• Total filters: 384\n'
        '• Total Params: ~3.35M\n'
        '• Fastest training time'
    )
    ax.text(0.5, 3, info_text, fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # CNN advantage box
    advantage_text = (
        'CNN Advantages:\n'
        '• Captures local patterns\n'
        '• Parallel computation\n'
        '• Position invariant\n'
        '• Fewer parameters'
    )
    ax.text(10.5, 3, advantage_text, fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('architecture_cnn.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: architecture_cnn.png")
    plt.close()


def create_comparison_figure():
    """Create a comparison figure showing all three architectures side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Model Architecture Comparison', fontsize=18, fontweight='bold', y=0.98)
    
    architectures = [
        ('LSTM', '#e74c3c'),
        ('GRU', '#e67e22'),
        ('CNN', '#f39c12')
    ]
    
    for idx, (ax, (name, color)) in enumerate(zip(axes, architectures)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, name, ha='center', fontsize=14, fontweight='bold')
        
        # Simple representation
        y = 8.5
        draw_box(ax, 5, y, 3, 0.6, 'Input', '#3498db', 'white')
        draw_arrow(ax, 5, y-0.4, 5, y-1.0)
        
        y = 7.3
        draw_box(ax, 5, y, 3, 0.6, 'Embedding', '#9b59b6', 'white')
        draw_arrow(ax, 5, y-0.4, 5, y-1.0)
        
        y = 6.1
        if name == 'CNN':
            # Show parallel structure for CNN
            draw_box(ax, 3.5, y, 1.8, 1.2, f'{name}\nParallel\nFilters', color, 'white')
            draw_box(ax, 6.5, y, 1.8, 1.2, f'{name}\nParallel\nFilters', color, 'white')
            draw_arrow(ax, 3.5, y-0.7, 5, y-1.8)
            draw_arrow(ax, 6.5, y-0.7, 5, y-1.8)
        else:
            draw_box(ax, 5, y, 3, 1.2, f'{name}\nLayers', color, 'white')
            draw_arrow(ax, 5, y-0.7, 5, y-1.0)
        
        y = 3.8
        draw_box(ax, 5, y, 2.5, 0.6, 'FC Layers', '#16a085', 'white')
        draw_arrow(ax, 5, y-0.4, 5, y-1.0)
        
        y = 2.6
        draw_box(ax, 5, y, 2.5, 0.6, 'Output', '#2ecc71', 'white')
        
        # Stats
        if name == 'LSTM':
            stats = 'Params: ~5.62M\nAUC: 82.21%\nBest for: Sequences'
        elif name == 'GRU':
            stats = 'Params: ~4.94M\nAUC: 85.19%\nBest for: Balance'
        else:
            stats = 'Params: ~3.35M\nAUC: 87.58%\nBest for: Speed'
        
        ax.text(5, 1, stats, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('architecture_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: architecture_comparison.png")
    plt.close()


def main():
    print("="*70)
    print("CREATING ARCHITECTURE FIGURES")
    print("="*70)
    
    print("\n[1/4] Creating LSTM architecture...")
    create_lstm_architecture()
    
    print("\n[2/4] Creating GRU architecture...")
    create_gru_architecture()
    
    print("\n[3/4] Creating CNN architecture...")
    create_cnn_architecture()
    
    print("\n[4/4] Creating comparison figure...")
    create_comparison_figure()
    
    print("\n" + "="*70)
    print("✓ All architecture figures created successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - architecture_lstm.png")
    print("  - architecture_gru.png")
    print("  - architecture_cnn.png")
    print("  - architecture_comparison.png")


if __name__ == "__main__":
    main()
