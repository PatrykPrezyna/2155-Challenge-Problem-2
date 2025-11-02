import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Economist color palette
ECONOMIST_RED = '#E3120B'
ECONOMIST_BLUE = '#00A0DC'
ECONOMIST_LIGHT_BLUE = '#6DCCF6'
ECONOMIST_LIGHT_GRAY = '#E8E8E8'
ECONOMIST_DARK_GRAY = '#5F6A72'
ECONOMIST_BLACK = '#1A1A1A'

# Read and process the CSV
df = pd.read_csv('results.csv')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Strip whitespace from string columns
df['Model group'] = df['Model group'].str.strip()
df['Model name'] = df['Model name'].str.strip()

# Remove empty rows
df = df.dropna(subset=['Model group', 'Model name'])

# Clean and convert numeric columns to handle malformed data
numeric_cols = [col for col in df.columns if 'R2 score' in col]
for col in numeric_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].str.replace(r'^0\.0\.', '0.', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Get unique model groups and advisors
model_groups = df['Model group'].unique()
advisors = [0, 1, 2, 3]

# Create figure with better proportions
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(5, len(model_groups), height_ratios=[0.15, 1, 1, 1, 1], 
                      hspace=0.45, wspace=0.35, 
                      left=0.07, right=0.97, top=0.92, bottom=0.06)

# Set overall figure style
fig.patch.set_facecolor('white')

# Add main title and subtitle in header area
title_ax = fig.add_subplot(gs[0, :])
title_ax.axis('off')
# title_ax.text(0, 0.65, 'Model performance analysis', 
#               fontsize=26, fontweight='bold', 
#               family='sans-serif', va='top',
#               color=ECONOMIST_BLACK)
# title_ax.text(0, 0.15, 'R² scores by advisor and model type', 
#               fontsize=12, color=ECONOMIST_DARK_GRAY, 
#               family='sans-serif', va='top')

# Add legend in header
legend_elements = [
    mpatches.Patch(facecolor=ECONOMIST_BLUE, alpha=0.5, label='Training set'),
    mpatches.Patch(facecolor=ECONOMIST_RED, label='Test set')
]
title_ax.legend(handles=legend_elements, loc='upper right', 
                bbox_to_anchor=(1.0, 0.8), frameon=False, 
                fontsize=11, ncol=2, columnspacing=1.5)

# Plot each combination
for row_idx, advisor in enumerate(advisors):
    for col_idx, model_group in enumerate(model_groups):
        ax = fig.add_subplot(gs[row_idx + 1, col_idx])
        
        # Filter data for this model group
        group_data = df[df['Model group'] == model_group].copy()
        
        if len(group_data) == 0:
            ax.axis('off')
            continue
        
        # Get train and test columns for this advisor
        train_col = f'Advisor {advisor} Train Set R2 score:'
        test_col = f'Advisor {advisor} Test Set R2 score:' if advisor < 3 else f'Advisor{advisor} Test Set R2 score:'
        
        # Extract values
        train_scores = group_data[train_col].values
        test_scores = group_data[test_col].values
        model_names = group_data['Model name'].values
        
        # Filter out NaN values
        valid_indices = ~(np.isnan(train_scores) | np.isnan(test_scores))
        train_scores = train_scores[valid_indices]
        test_scores = test_scores[valid_indices]
        model_names = model_names[valid_indices]
        
        if len(train_scores) == 0:
            ax.axis('off')
            continue
        
        # Set up x positions with better spacing
        x = np.arange(len(model_names))
        width = 0.38
        
        # Plot bars with rounded corners effect
        bars1 = ax.bar(x - width/2, train_scores, width, 
                      color=ECONOMIST_BLUE, alpha=0.5, 
                      edgecolor='none', linewidth=0,
                      zorder=3)
        bars2 = ax.bar(x + width/2, test_scores, width, 
                      color=ECONOMIST_RED, 
                      edgecolor='none', linewidth=0,
                      zorder=3)
        
        # Add value labels on top of bars
        for i, (train_val, test_val) in enumerate(zip(train_scores, test_scores)):
            # Training bar label
            if not np.isnan(train_val):
                ax.text(x[i] - width/2, train_val, f'{train_val:.2f}', 
                       ha='center', va='bottom', fontsize=7,
                       color=ECONOMIST_DARK_GRAY, fontweight='500')
            
            # Test bar label
            if not np.isnan(test_val):
                ax.text(x[i] + width/2, test_val, f'{test_val:.2f}', 
                       ha='center', va='bottom', fontsize=7,
                       color=ECONOMIST_DARK_GRAY, fontweight='500')
        
        # Only show title elements on appropriate edges
        # Advisor name only on leftmost column
        if col_idx == 0:
            ax.set_ylabel(f'Advisor {advisor}', 
                         fontsize=11, color=ECONOMIST_DARK_GRAY,
                         labelpad=10, family='sans-serif', fontweight='600')
        else:
            ax.set_ylabel('')
        
        # Model group name only on top row
        if row_idx == 0:
            ax.set_title(f'{model_group}', 
                        fontsize=12, fontweight='600', pad=12,
                        color=ECONOMIST_BLACK, family='sans-serif')
        
        # X-axis labels only on bottom row
        ax.set_xticks(x)
        if row_idx == len(advisors) - 1:
            ax.set_xticklabels(model_names, rotation=45, ha='right', 
                              fontsize=8.5, color=ECONOMIST_DARK_GRAY,
                              family='sans-serif')
        else:
            ax.set_xticklabels([])
        
        # Y-axis styling
        ax.tick_params(axis='y', labelsize=9, colors=ECONOMIST_DARK_GRAY,
                      length=0, pad=6)
        ax.tick_params(axis='x', length=0, pad=4)
        
        # Enhanced grid with lighter color
        ax.yaxis.grid(True, color=ECONOMIST_LIGHT_GRAY, linewidth=0.6, 
                     zorder=1, alpha=0.7)
        ax.set_axisbelow(True)
        
        # Background color for alternating effect (subtle)
        ax.set_facecolor('#FAFAFA')
        
        # Spines styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(ECONOMIST_LIGHT_GRAY)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_color(ECONOMIST_DARK_GRAY)
        ax.spines['left'].set_linewidth(1.5)
        
        # Set y-axis limits with better padding
        if len(train_scores) > 0 and len(test_scores) > 0:
            y_min = min(0, np.nanmin(train_scores), np.nanmin(test_scores))
            y_max = max(np.nanmax(train_scores), np.nanmax(test_scores))
            padding = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - padding, y_max + padding)
        
        # Add subtle shadow effect to bars
        for bar in bars2:
            bar.set_linewidth(0.5)

# Add source note with better styling
# fig.text(0.07, 0.02, 'Source: Model evaluation results', 
#          fontsize=9, style='italic', color=ECONOMIST_DARK_GRAY,
#          family='sans-serif', alpha=0.8)

# Add a subtle border around the entire figure
# rect = mpatches.Rectangle((0.06, 0.05), 0.92, 0.88, 
#                           linewidth=1.5, edgecolor=ECONOMIST_LIGHT_GRAY, 
#                           facecolor='none', transform=fig.transFigure, 
#                           clip_on=False, zorder=1000)
# fig.patches.append(rect)

# Save with high quality
plt.savefig('model_performance_economist_style.png', dpi=300, 
            bbox_inches='tight', facecolor='white', edgecolor='none',
            pad_inches=0.1)

# Display
plt.show()

print("✓ Visualization created successfully!")
print(f"✓ Saved as 'model_performance_economist_style.png'")