"""
2D visualization dashboard for psychedelic compounds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("Set2")

def setup_output_dir():
    """Create outputs directory if it doesn't exist."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def plot_dashboard(df: pd.DataFrame, save_path: str = None) -> str:
    """
    Create a 4-panel dashboard visualization.
    
    Args:
        df: DataFrame with molecular descriptors
        save_path: Optional path to save the plot
        
    Returns:
        Path to the saved plot
    """
    if save_path is None:
        output_dir = setup_output_dir()
        save_path = output_dir / "dashboard.png"
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Psychedelic Therapeutics Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Panel 1: Molecular Weight vs LogP colored by class
    ax1 = axes[0, 0]
    
    # Get unique classes and colors
    classes = df['class'].unique()
    colors = sns.color_palette("Set2", n_colors=len(classes))
    class_colors = dict(zip(classes, colors))
    
    for compound_class in classes:
        class_data = df[df['class'] == compound_class]
        ax1.scatter(class_data['mw'], class_data['logp'], 
                   label=compound_class, alpha=0.7, s=80,
                   color=class_colors[compound_class])
    
    ax1.set_xlabel('Molecular Weight (Da)')
    ax1.set_ylabel('LogP')
    ax1.set_title('Chemical Space: LogP vs Molecular Weight')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: TPSA histogram
    ax2 = axes[0, 1]
    
    # Filter out NaN values for histogram
    tpsa_values = df['tpsa'].dropna()
    
    if len(tpsa_values) > 0:
        ax2.hist(tpsa_values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=60, color='red', linestyle='--', alpha=0.7, 
                   label='BBB threshold (60 Ų)')
        ax2.legend()
    
    ax2.set_xlabel('TPSA (Ų)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Topological Polar Surface Area Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Average drug-likeness by class
    ax3 = axes[1, 0]
    
    # Calculate average drug-likeness by class
    class_drug_likeness = df.groupby('class')['drug_likeness'].mean().sort_values(ascending=False)
    
    bars = ax3.bar(range(len(class_drug_likeness)), class_drug_likeness.values,
                   color=[class_colors[cls] for cls in class_drug_likeness.index])
    
    ax3.set_xlabel('Compound Class')
    ax3.set_ylabel('Average Drug-likeness Score')
    ax3.set_title('Drug-likeness by Compound Class')
    ax3.set_xticks(range(len(class_drug_likeness)))
    ax3.set_xticklabels(class_drug_likeness.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, class_drug_likeness.values)):
        ax3.text(bar.get_x() + bar.get_width()/2., value + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: BBB penetration pie chart
    ax4 = axes[1, 1]
    
    bbb_counts = df['bbb_label'].value_counts()
    
    if len(bbb_counts) > 0:
        colors_pie = ['#6BCF7F', '#FFD93D', '#FF6B6B']  # Good, Unknown, Poor
        wedges, texts, autotexts = ax4.pie(bbb_counts.values, 
                                          labels=bbb_counts.index,
                                          autopct='%1.1f%%', 
                                          colors=colors_pie[:len(bbb_counts)])
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
    
    ax4.set_title('Blood-Brain Barrier Penetration Prediction')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close to free memory
    
    print(f"Dashboard saved to: {save_path}")
    return str(save_path)

def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None) -> str:
    """
    Create a correlation heatmap of molecular descriptors.
    
    Args:
        df: DataFrame with molecular descriptors
        save_path: Optional path to save the plot
        
    Returns:
        Path to the saved plot
    """
    if save_path is None:
        output_dir = setup_output_dir()
        save_path = output_dir / "correlation_heatmap.png"
    
    # Select numeric descriptor columns
    descriptor_cols = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotb', 'rings', 'drug_likeness']
    available_cols = [col for col in descriptor_cols if col in df.columns]
    
    if len(available_cols) < 2:
        print("Not enough numeric columns for correlation analysis")
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[available_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Molecular Descriptor Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Correlation heatmap saved to: {save_path}")
    return str(save_path)

def plot_class_comparison(df: pd.DataFrame, save_path: str = None) -> str:
    """
    Create detailed comparison plots by compound class.
    
    Args:
        df: DataFrame with molecular descriptors
        save_path: Optional path to save the plot
        
    Returns:
        Path to the saved plot
    """
    if save_path is None:
        output_dir = setup_output_dir()
        save_path = output_dir / "class_comparison.png"
    
    # Select key descriptors for comparison
    key_descriptors = ['mw', 'logp', 'tpsa', 'drug_likeness']
    available_descriptors = [col for col in key_descriptors if col in df.columns]
    
    if len(available_descriptors) < 2:
        print("Not enough descriptors for class comparison")
        return None
    
    # Create subplots
    n_descriptors = len(available_descriptors)
    cols = 2
    rows = (n_descriptors + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    fig.suptitle('Molecular Properties by Compound Class', fontsize=16, fontweight='bold')
    
    for i, descriptor in enumerate(available_descriptors):
        ax = axes[i]
        
        # Box plot by class
        df.boxplot(column=descriptor, by='class', ax=ax)
        ax.set_title(f'{descriptor.upper()} by Class')
        ax.set_xlabel('Compound Class')
        ax.set_ylabel(descriptor.upper())
        
        # Rotate x-axis labels if needed
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(len(available_descriptors), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Class comparison plot saved to: {save_path}")
    return str(save_path)

if __name__ == "__main__":
    # Test visualization
    from data import load_demo
    from descriptors import compute
    
    print("Loading demo data...")
    df = load_demo()
    
    print("Computing descriptors...")
    df_with_descriptors = compute(df)
    
    print("Creating dashboard...")
    dashboard_path = plot_dashboard(df_with_descriptors)
    
    print("Creating correlation heatmap...")
    heatmap_path = plot_correlation_heatmap(df_with_descriptors)
    
    print("Creating class comparison...")
    comparison_path = plot_class_comparison(df_with_descriptors)
    
    print("All visualizations completed successfully!")