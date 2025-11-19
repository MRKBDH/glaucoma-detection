"""
Create improved architecture diagrams for LSTM, GRU, and CNN models
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9


def draw_box(ax, x, y, width, height, text, color, text_color='white', edge_width=2):
    """Draw a colored box with text"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.05", 
        facecolor=color, 
        edgecolor='black', 
        linewidth=edge_width
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', 
            fontsize=9, fontweight='bold', color=text_color)


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


def create_comparison_figure():
    """Create an improved comparison figure"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle('Deep Learning Architectures for Glaucoma Detection from Clinical Notes', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # LSTM Architecture
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.text(5, 11.2, 'LSTM (Long Short-Term Memory)', ha='center', 
            fontsize=13, fontweight='bold', color='#c0392b')
    
    y = 10
    draw_box(ax, 5, y, 4, 0.7, 'Input: Clinical Notes', '#3498db', 'white')
    ax.text(7.5, y, 'Text', ha='left', fontsize=7, style='italic')
    draw_arrow(ax, 5, y-0.4, 5, y-0.9)
    
    y = 8.8
    draw_box(ax, 5, y, 4, 0.7, 'Text Preprocessing', '#95a5a6', 'white')
    ax.text(7.5, y+0.2, 'Tokenization', ha='left', fontsize=7, style='italic')
    ax.text(7.5, y-0.2, 'Cleaning', ha='left', fontsize=7, style='italic')
    draw_arrow(ax, 5, y-0.4, 5, y-0.9)
    
    y = 7.6
    draw_box(ax, 5, y, 4, 0.7, 'Embedding Layer (300D)', '#9b59b6', 'white')
    ax.text(7.5, y, 'Word→Vector', ha='left', fontsize=7, style='italic')
    draw_arrow(ax, 5, y-0.4, 5, y-0.9)
    
    y = 6.2
    draw_box(ax, 5, y, 4.5, 1.0, 'Bidirectional LSTM Layer 1\nHidden: 256 (×2 = 512)', 
             '#e74c3c', 'white')
    # Draw small LSTM cell
    cell_x, cell_y = 2.2, 6.2
    rect = Rectangle((cell_x-0.25, cell_y-0.35), 0.5, 0.7, 
                     facecolor='#c0392b', edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(cell_x, cell_y+0.15, 'h', ha='center', fontsize=10, 
            fontweight='bold', color='white')
    ax.text(cell_x, cell_y-0.15, 'c', ha='center', fontsize=10, 
            fontweight='bold', color='white')
    ax.text(cell_x-0.7, cell_y, 'Cell\nState', ha='center', fontsize=6)
    draw_arrow(ax, 5, y-0.6, 5, y-1.2)
    
    y = 4.7
    draw_box(ax, 5, y, 3, 0.5, 'Dropout (0.3)', '#7f8c8d', 'white')
    draw_arrow(ax, 5, y-0.3, 5, y-0.8)
    
    y = 3.6
    draw_box(ax, 5, y, 4.5, 1.0, 'Bidirectional LSTM Layer 2\nHidden: 256 (×2 = 512)', 
             '#e74c3c', 'white')
    draw_arrow(ax, 5, y-0.6, 5, y-1.2)
    
    y = 2.1
    draw_box(ax, 5, y, 3, 0.5, 'Dropout (0.3)', '#7f8c8d', 'white')
    draw_arrow(ax, 5, y-0.3, 5, y-0.8)
    
    y = 1.0
    draw_box(ax, 5, y, 3.5, 0.6, 'FC (512→128) + ReLU + Dropout', '#16a085', 'white')
    draw_arrow(ax, 5, y-0.4, 5, y-0.8)
    
    y = 0.0
    draw_box(ax, 5, y, 3, 0.6, 'Output (Sigmoid)', '#2ecc71', 'white')
    ax.text(7, y, 'P(Glaucoma)', ha='left', fontsize=7, style='italic')
    
    # Stats box
    stats_text = 'Parameters: ~5.62M\nAUC: 82.21%\nSensitivity: 75.46%\nSpecificity: 74.21%'
    ax.text(5, -1.0, stats_text, ha='center', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe4e1', alpha=0.8, edgecolor='black'))
    
    ax.text(5, -1.8, '✓ Best for sequential patterns\n✓ Captures long-term dependencies', 
            ha='center', fontsize=7, style='italic')
    
    # GRU Architecture
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.text(5, 11.2, 'GRU (Gated Recurrent Unit)', ha='center', 
            fontsize=13, fontweight='bold', color='#d35400')
    
    y = 10
    draw_box(ax, 5, y, 4, 0.7, 'Input: Clinical Notes', '#3498db', 'white')
    ax.text(7.5, y, 'Text', ha='left', fontsize=7, style='italic')
    draw_arrow(ax, 5, y-0.4, 5, y-0.9)
    
    y = 8.8
    draw_box(ax, 5, y, 4, 0.7, 'Text Preprocessing', '#95a5a6', 'white')
    ax.text(7.5, y+0.2, 'Tokenization', ha='left', fontsize=7, style='italic')
    ax.text(7.5, y-0.2, 'Cleaning', ha='left', fontsize=7, style='italic')
    draw_arrow(ax, 5, y-0.4, 5, y-0.9)
    
    y = 7.6
    draw_box(ax, 5, y, 4, 0.7, 'Embedding Layer (300D)', '#9b59b6', 'white')
    ax.text(7.5, y, 'Word→Vector', ha='left', fontsize=7, style='italic')
    draw_arrow(ax, 5, y-0.4, 5, y-0.9)
    
    y = 6.2
    draw_box(ax, 5, y, 4.5, 1.0, 'Bidirectional GRU Layer 1\nHidden: 256 (×2 = 512)', 
             '#e67e22', 'white')
    # Draw small GRU cell
    cell_x, cell_y = 2.2, 6.2
    rect = Rectangle((cell_x-0.25, cell_y-0.35), 0.5, 0.7, 
                     facecolor='#d35400', edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(cell_x, cell_y, 'h', ha='center', fontsize=12, 
            fontweight='bold', color='white')
    ax.text(cell_x-0.7, cell_y, 'Hidden\nOnly', ha='center', fontsize=6)
    draw_arrow(ax, 5, y-0.6, 5, y-1.2)
    
    y = 4.7
    draw_box(ax, 5, y, 3, 0.5, 'Dropout (0.3)', '#7f8c8d', 'white')
    draw_arrow(ax, 5, y-0.3, 5, y-0.8)
    
    y = 3.6
    draw_box(ax, 5, y, 4.5, 1.0, 'Bidirectional GRU Layer 2\nHidden: 256 (×2 = 512)', 
             '#e67e22', 'white')
    draw_arrow(ax, 5, y-0.6, 5, y-1.2)
    
    y = 2.1
    draw_box(ax, 5, y, 3, 0.5, 'Dropout (0.3)', '#7f8c8d', 'white')
    draw_arrow(ax, 5, y-0.3, 5, y-0.8)
    
    y = 1.0
    draw_box(ax, 5, y, 3.5, 0.6, 'FC (512→128) + ReLU + Dropout', '#16a085', 'white')
    draw_arrow(ax, 5, y-0.4, 5, y-0.8)
    
    y = 0.0
    draw_box(ax, 5, y, 3, 0.6, 'Output (Sigmoid)', '#2ecc71', 'white')
    ax.text(7, y, 'P(Glaucoma)', ha='left', fontsize=7, style='italic')
    
    # Stats box
    stats_text = 'Parameters: ~4.94M\nAUC: 85.19%\nSensitivity: 79.18%\nSpecificity: 73.90%'
    ax.text(5, -1.0, stats_text, ha='center', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd', alpha=0.8, edgecolor='black'))
    
    ax.text(5, -1.8, '✓ Faster than LSTM (fewer gates)\n✓ Good balance of performance', 
            ha='center', fontsize=7, style='italic')
    
    # CNN Architecture
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.text(5, 11.2, '1D CNN (Convolutional Neural Network)', ha='center', 
            fontsize=13, fontweight='bold', color='#d68910')
    
    y = 10
    draw_box(ax, 5, y, 4, 0.7, 'Input: Clinical Notes', '#3498db', 'white')
    ax.text(7.5, y, 'Text', ha='left', fontsize=7, style='italic')
    draw_arrow(ax, 5, y-0.4, 5, y-0.9)
    
    y = 8.8
    draw_box(ax, 5, y, 4, 0.7, 'Text Preprocessing', '#95a5a6', 'white')
    ax.text(7.5, y+0.2, 'Tokenization', ha='left', fontsize=7, style='italic')
    ax.text(7.5, y-0.2, 'Cleaning', ha='left', fontsize=7, style='italic')
    draw_arrow(ax, 5, y-0.4, 5, y-0.9)
    
    y = 7.6
    draw_box(ax, 5, y, 4, 0.7, 'Embedding Layer (300D)', '#9b59b6', 'white')
    ax.text(7.5, y, 'Word→Vector', ha='left', fontsize=7, style='italic')
    
    # Split into 3 parallel paths
    draw_arrow(ax, 5, y-0.4, 3, y-1.1)
    draw_arrow(ax, 5, y-0.4, 5, y-1.1)
    draw_arrow(ax, 5, y-0.4, 7, y-1.1)
    
    # Parallel convolutions
    y = 5.8
    draw_box(ax, 3, y, 1.6, 1.0, 'Conv1D\nK=3\nF=128', '#f39c12', 'white')
    draw_box(ax, 5, y, 1.6, 1.0, 'Conv1D\nK=4\nF=128', '#f39c12', 'white')
    draw_box(ax, 7, y, 1.6, 1.0, 'Conv1D\nK=5\nF=128', '#f39c12', 'white')
    
    # Visual filters
    for i, x_pos in enumerate([3, 5, 7]):
        for j in range(3):
            rect_y = y + 0.25 - j*0.12
            ax.add_patch(Rectangle((x_pos-0.3+j*0.1, rect_y), 0.35, 0.06, 
                                  facecolor='yellow', edgecolor='black', 
                                  linewidth=0.5, alpha=0.7))
    
    ax.text(3, y-0.7, '↓', ha='center', fontsize=16)
    ax.text(5, y-0.7, '↓', ha='center', fontsize=16)
    ax.text(7, y-0.7, '↓', ha='center', fontsize=16)
    
    y = 4.4
    draw_box(ax, 3, y, 1.6, 0.5, 'MaxPool', '#3498db', 'white')
    draw_box(ax, 5, y, 1.6, 0.5, 'MaxPool', '#3498db', 'white')
    draw_box(ax, 7, y, 1.6, 0.5, 'MaxPool', '#3498db', 'white')
    
    # Converge
    draw_arrow(ax, 3, y-0.3, 5, y-1.0)
    draw_arrow(ax, 5, y-0.3, 5, y-1.0)
    draw_arrow(ax, 7, y-0.3, 5, y-1.0)
    
    y = 3.0
    draw_box(ax, 5, y, 3.5, 0.6, 'Concatenate (384 features)', '#8e44ad', 'white')
    draw_arrow(ax, 5, y-0.4, 5, y-0.8)
    
    y = 2.1
    draw_box(ax, 5, y, 3, 0.5, 'Dropout (0.3)', '#7f8c8d', 'white')
    draw_arrow(ax, 5, y-0.3, 5, y-0.8)
    
    y = 1.0
    draw_box(ax, 5, y, 3.5, 0.6, 'FC (384→128) + ReLU + Dropout', '#16a085', 'white')
    draw_arrow(ax, 5, y-0.4, 5, y-0.8)
    
    y = 0.0
    draw_box(ax, 5, y, 3, 0.6, 'Output (Sigmoid)', '#2ecc71', 'white')
    ax.text(7, y, 'P(Glaucoma)', ha='left', fontsize=7, style='italic')
    
    # Stats box
    stats_text = 'Parameters: ~3.35M\nAUC: 87.58%\nSensitivity: 89.15%\nSpecificity: 66.84%'
    ax.text(5, -1.0, stats_text, ha='center', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#d4edda', alpha=0.8, edgecolor='black'))
    
    ax.text(5, -1.8, '✓ Fastest training & inference\n✓ Captures local n-gram patterns', 
            ha='center', fontsize=7, style='italic')
    
    plt.tight_layout()
    plt.savefig('architecture_comparison_improved.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: architecture_comparison_improved.png")
    plt.close()


def create_detailed_cnn():
    """Create detailed CNN architecture"""
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 13)
    ax.axis('off')
    
    ax.text(7, 12.5, '1D Convolutional Neural Network for Clinical Text Classification', 
            ha='center', fontsize=15, fontweight='bold')
    
    y = 11.2
    draw_box(ax, 7, y, 4, 0.8, 'Input: Clinical Notes (Text)', '#3498db', 'white')
    ax.text(10, y, 'Variable length', ha='left', fontsize=8, style='italic')
    draw_arrow(ax, 7, y-0.5, 7, y-1.1)
    
    y = 9.8
    draw_box(ax, 7, y, 4, 0.8, 'Text Preprocessing & Tokenization', '#95a5a6', 'white')
    ax.text(10, y, 'Clean & tokenize', ha='left', fontsize=8, style='italic')
    draw_arrow(ax, 7, y-0.5, 7, y-1.1)
    
    y = 8.4
    draw_box(ax, 7, y, 4, 0.8, 'Embedding Layer (300-dimensional)', '#9b59b6', 'white')
    ax.text(10, y, 'Shape: [batch, seq, 300]', ha='left', fontsize=8, style='italic')
    
    # Split to parallel convolutions
    draw_arrow(ax, 7, y-0.5, 3.5, y-1.3)
    draw_arrow(ax, 7, y-0.5, 7, y-1.3)
    draw_arrow(ax, 7, y-0.5, 10.5, y-1.3)
    
    # Three parallel convolution branches
    y = 6.2
    
    # Branch 1
    draw_box(ax, 3.5, y, 2.5, 1.2, 'Conv1D\nKernel Size: 3\nFilters: 128\n+ ReLU', 
             '#f39c12', 'white')
    for j in range(4):
        rect_y = y + 0.3 - j*0.15
        ax.add_patch(Rectangle((3.5-0.5+j*0.15, rect_y), 0.6, 0.08, 
                              facecolor='yellow', edgecolor='black', linewidth=0.5, alpha=0.7))
    ax.text(1.8, y, 'Captures\n3-grams', ha='center', fontsize=7, style='italic')
    ax.text(3.5, y-0.8, '↓', ha='center', fontsize=18)
    draw_box(ax, 3.5, y-1.5, 2.2, 0.6, 'MaxPooling1D', '#3498db', 'white')
    
    # Branch 2
    draw_box(ax, 7, y, 2.5, 1.2, 'Conv1D\nKernel Size: 4\nFilters: 128\n+ ReLU', 
             '#f39c12', 'white')
    for j in range(5):
        rect_y = y + 0.3 - j*0.15
        ax.add_patch(Rectangle((7-0.5+j*0.12, rect_y), 0.5, 0.08, 
                              facecolor='yellow', edgecolor='black', linewidth=0.5, alpha=0.7))
    ax.text(5.3, y, 'Captures\n4-grams', ha='center', fontsize=7, style='italic')
    ax.text(7, y-0.8, '↓', ha='center', fontsize=18)
    draw_box(ax, 7, y-1.5, 2.2, 0.6, 'MaxPooling1D', '#3498db', 'white')
    
    # Branch 3
    draw_box(ax, 10.5, y, 2.5, 1.2, 'Conv1D\nKernel Size: 5\nFilters: 128\n+ ReLU', 
             '#f39c12', 'white')
    for j in range(6):
        rect_y = y + 0.3 - j*0.15
        ax.add_patch(Rectangle((10.5-0.5+j*0.1, rect_y), 0.45, 0.08, 
                              facecolor='yellow', edgecolor='black', linewidth=0.5, alpha=0.7))
    ax.text(12.2, y, 'Captures\n5-grams', ha='center', fontsize=7, style='italic')
    ax.text(10.5, y-0.8, '↓', ha='center', fontsize=18)
    draw_box(ax, 10.5, y-1.5, 2.2, 0.6, 'MaxPooling1D', '#3498db', 'white')
    
    # Merge branches
    draw_arrow(ax, 3.5, 4.0, 7, 3.0)
    draw_arrow(ax, 7, 4.0, 7, 3.0)
    draw_arrow(ax, 10.5, 4.0, 7, 3.0)
    
    y = 2.5
    draw_box(ax, 7, y, 4, 0.8, 'Concatenate: 128 + 128 + 128 = 384', '#8e44ad', 'white')
    draw_arrow(ax, 7, y-0.5, 7, y-1.1)
    
    y = 1.0
    draw_box(ax, 7, y, 3, 0.6, 'Dropout (0.3)', '#7f8c8d', 'white')
    draw_arrow(ax, 7, y-0.4, 7, y-0.9)
    
    y = -0.2
    draw_box(ax, 7, y, 4, 0.7, 'Fully Connected (384 → 128) + ReLU', '#16a085', 'white')
    draw_arrow(ax, 7, y-0.5, 7, y-1.0)
    
    y = -1.5
    draw_box(ax, 7, y, 3, 0.6, 'Dropout (0.3)', '#7f8c8d', 'white')
    draw_arrow(ax, 7, y-0.4, 7, y-0.9)
    
    y = -2.7
    draw_box(ax, 7, y, 4, 0.7, 'Fully Connected (128 → 1) + Sigmoid', '#2ecc71', 'white')
    ax.text(10, y, 'P(Glaucoma)', ha='left', fontsize=9, style='italic')
    
    # Info boxes
    info1 = ('Advantages:\n'
             '• Parallel computation\n'
             '• Position invariant\n'
             '• Fewer parameters\n'
             '• Fast training')
    ax.text(1.5, 2, info1, fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4, edgecolor='black'))
    
    info2 = ('Performance:\n'
             '• Total Params: 3.35M\n'
             '• AUC: 87.58%\n'
             '• Sensitivity: 89.15%\n'
             '• Specificity: 66.84%')
    ax.text(11.5, 2, info2, fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig('architecture_cnn_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: architecture_cnn_detailed.png")
    plt.close()


def main():
    print("="*70)
    print("CREATING IMPROVED ARCHITECTURE FIGURES")
    print("="*70)
    
    print("\n[1/2] Creating improved comparison figure...")
    create_comparison_figure()
    
    print("\n[2/2] Creating detailed CNN architecture...")
    create_detailed_cnn()
    
    print("\n" + "="*70)
    print("✓ Improved architecture figures created!")
    print("="*70)
    print("\nGenerated files:")
    print("  - architecture_comparison_improved.png")
    print("  - architecture_cnn_detailed.png")


if __name__ == "__main__":
    main()
