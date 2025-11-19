"""
Compare all three models: LSTM, GRU, and CNN
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Results data
models = ['LSTM', 'GRU', 'CNN']

# Overall metrics
overall_auc = [0.8221, 0.8519, 0.8758]
overall_sens = [0.7546, 0.7918, 0.8915]
overall_spec = [0.7421, 0.7390, 0.6684]

# By race
white_auc = [0.8136, 0.8391, 0.8617]
black_auc = [0.8445, 0.8855, 0.9063]
asian_auc = [0.8713, 0.9184, 0.9367]

# 1. Overall AUC Comparison
plt.figure(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = plt.bar(models, overall_auc, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.ylabel('AUC Score', fontsize=13, fontweight='bold')
plt.xlabel('Model', fontsize=13, fontweight='bold')
plt.title('Overall AUC Comparison', fontsize=15, fontweight='bold')
plt.ylim([0.75, 0.90])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/overall_auc_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/overall_auc_comparison.png")
plt.close()

# 2. Sensitivity vs Specificity
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, overall_sens, width, label='Sensitivity', 
               color='#FF6B6B', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, overall_spec, width, label='Specificity', 
               color='#4ECDC4', alpha=0.8, edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_xlabel('Model', fontsize=13, fontweight='bold')
ax.set_title('Sensitivity vs Specificity Comparison', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=12)
ax.set_ylim([0.6, 0.95])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/sens_spec_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/sens_spec_comparison.png")
plt.close()

# 3. Performance by Race
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, white_auc, width, label='White', 
              color='#1f77b4', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, black_auc, width, label='Black', 
              color='#ff7f0e', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, asian_auc, width, label='Asian', 
              color='#2ca02c', alpha=0.8, edgecolor='black')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('AUC Score', fontsize=13, fontweight='bold')
ax.set_xlabel('Model', fontsize=13, fontweight='bold')
ax.set_title('AUC Performance by Demographic Group', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=12)
ax.set_ylim([0.75, 0.96])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/auc_by_race.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/auc_by_race.png")
plt.close()

# 4. Heatmap of all metrics
metrics_data = np.array([
    overall_auc,
    overall_sens,
    overall_spec,
    white_auc,
    black_auc,
    asian_auc
])

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(metrics_data, cmap='YlOrRd', aspect='auto', vmin=0.65, vmax=0.95)

# Labels
row_labels = ['Overall AUC', 'Sensitivity', 'Specificity', 
              'White AUC', 'Black AUC', 'Asian AUC']
col_labels = models

ax.set_xticks(np.arange(len(col_labels)))
ax.set_yticks(np.arange(len(row_labels)))
ax.set_xticklabels(col_labels, fontsize=12)
ax.set_yticklabels(row_labels, fontsize=11)

# Add values
for i in range(len(row_labels)):
    for j in range(len(col_labels)):
        text = ax.text(j, i, f'{metrics_data[i, j]:.4f}',
                      ha="center", va="center", color="black", 
                      fontweight='bold', fontsize=11)

ax.set_title('Complete Performance Heatmap', fontsize=15, fontweight='bold', pad=20)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/performance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/performance_heatmap.png")
plt.close()

# 5. Create summary table
with open('figures/final_results_table.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("           FINAL RESULTS - ALL THREE MODELS\n")
    f.write("="*80 + "\n\n")
    
    f.write("OVERALL PERFORMANCE\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Model':<10} {'AUC':<12} {'Sensitivity':<15} {'Specificity':<15}\n")
    f.write("-"*80 + "\n")
    for i, model in enumerate(models):
        f.write(f"{model:<10} {overall_auc[i]:<12.4f} {overall_sens[i]:<15.4f} {overall_spec[i]:<15.4f}\n")
    
    f.write("\n\nPERFORMANCE BY DEMOGRAPHIC GROUP (AUC)\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Model':<10} {'White':<12} {'Black':<12} {'Asian':<12}\n")
    f.write("-"*80 + "\n")
    for i, model in enumerate(models):
        f.write(f"{model:<10} {white_auc[i]:<12.4f} {black_auc[i]:<12.4f} {asian_auc[i]:<12.4f}\n")
    
    f.write("\n\nKEY FINDINGS\n")
    f.write("-"*80 + "\n")
    f.write("1. Best Overall Model: CNN (87.58% AUC)\n")
    f.write("2. Best Sensitivity: CNN (89.15%)\n")
    f.write("3. Best Demographic Performance: CNN on Asian (93.67%)\n")
    f.write("4. Most Balanced: GRU (good AUC with balanced sens/spec)\n")
    f.write("5. All models show good fairness (>81% for all groups)\n")
    f.write("\n" + "="*80 + "\n")

print("✓ Saved: figures/final_results_table.txt")

print("\n" + "="*60)
print("✓ All comparison visualizations created!")
print("="*60)
