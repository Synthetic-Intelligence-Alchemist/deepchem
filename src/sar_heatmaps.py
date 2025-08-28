"""
SAR Heatmaps and Visual Analytics for Psychedelic Therapeutics
==============================================================

Advanced visualization tools for structure-activity relationships:
- Property correlation heatmaps
- Activity landscape analysis
- Chemical space visualization
- SAR cliff detection

Author: AI Assistant for CNS Therapeutics Research
Focus: Visual SAR analysis for 5-HT2A receptor ligands
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import DataStructs, Descriptors
    from rdkit.Chem.Fingerprints import FingerprintMols
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    RDKIT_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    print("⚠️ Some dependencies not available. Limited functionality.")

class SARVisualizer:
    """Advanced SAR visualization and analysis."""
    
    def __init__(self):
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Color schemes for different activity levels
        self.activity_colors = {
            'high': '#d62728',      # Red
            'moderate': '#ff7f0e',  # Orange  
            'low': '#2ca02c',       # Green
            'inactive': '#7f7f7f'   # Gray
        }
        
        # Property importance weights for 5-HT2A activity
        self.property_weights = {
            'mw': 0.15,
            'logp': 0.25,
            'tpsa': 0.20,
            'hbd': 0.10,
            'hba': 0.10,
            'rotb': 0.05,
            'rings': 0.15
        }
    
    def create_property_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Create enhanced property correlation heatmap."""
        # Select molecular properties
        prop_cols = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotb', 'rings']
        available_cols = [col for col in prop_cols if col in df.columns]
        
        if len(available_cols) < 3:
            print("Insufficient molecular properties for correlation analysis")
            return None
        
        # Add activity columns if available
        activity_cols = ['drug_likeness', 'bbb_score', 'ht2a_affinity_pred', 'cns_mpo']
        for col in activity_cols:
            if col in df.columns:
                available_cols.append(col)
        
        # Calculate correlation matrix
        corr_matrix = df[available_cols].corr()
        
        # Create enhanced heatmap
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   fmt='.2f')
        
        plt.title('Molecular Property Correlation Matrix\nPsychedelic Therapeutics Dataset', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "enhanced_correlation_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def create_activity_landscape(self, df: pd.DataFrame) -> str:
        """Create 3D activity landscape visualization."""
        if not SKLEARN_AVAILABLE or len(df) < 5:
            return None
        
        # Prepare data
        prop_cols = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotb']
        available_cols = [col for col in prop_cols if col in df.columns]
        
        if len(available_cols) < 3:
            return None
        
        # Get activity data
        activity_col = 'ht2a_affinity_pred' if 'ht2a_affinity_pred' in df.columns else 'drug_likeness'
        
        # Perform PCA for dimensionality reduction
        X = df[available_cols].fillna(0)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=X_pca[:, 0],
            y=X_pca[:, 1], 
            z=X_pca[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=df[activity_col],
                colorscale='Viridis',
                colorbar=dict(title=f"{activity_col.replace('_', ' ').title()}"),
                showscale=True
            ),
            text=[f"{row['name']}<br>{activity_col}: {row[activity_col]:.2f}" 
                  for _, row in df.iterrows()],
            hovertemplate='<b>%{text}</b><br>' +
                         'PC1: %{x:.2f}<br>' +
                         'PC2: %{y:.2f}<br>' +
                         'PC3: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='3D Activity Landscape<br>PCA of Molecular Properties',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)'
            ),
            width=900,
            height=700
        )
        
        # Save plot
        output_path = self.output_dir / "activity_landscape_3d.html"
        fig.write_html(output_path)
        
        return str(output_path)
    
    def create_chemical_space_map(self, df: pd.DataFrame) -> str:
        """Create chemical space visualization with t-SNE."""
        if not RDKIT_AVAILABLE or not SKLEARN_AVAILABLE or len(df) < 5:
            return None
        
        # Calculate molecular fingerprints
        fingerprints = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol:
                    fp = FingerprintMols.FingerprintMol(mol)
                    arr = np.zeros((1,))
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    fingerprints.append(arr)
                    valid_indices.append(idx)
            except:
                continue
        
        if len(fingerprints) < 5:
            return None
        
        # Perform t-SNE
        fingerprints_array = np.array(fingerprints)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(fingerprints)-1))
        X_tsne = tsne.fit_transform(fingerprints_array)
        
        # Create DataFrame for plotting
        plot_df = df.iloc[valid_indices].copy()
        plot_df['tsne_1'] = X_tsne[:, 0]
        plot_df['tsne_2'] = X_tsne[:, 1]
        
        # Create interactive plot
        fig = px.scatter(
            plot_df,
            x='tsne_1',
            y='tsne_2',
            color='class',
            size='mw' if 'mw' in plot_df.columns else None,
            hover_data=['name', 'smiles'],
            title='Chemical Space Map (t-SNE of Molecular Fingerprints)',
            labels={'tsne_1': 't-SNE Component 1', 'tsne_2': 't-SNE Component 2'}
        )
        
        fig.update_layout(
            width=900,
            height=700,
            showlegend=True
        )
        
        # Save plot
        output_path = self.output_dir / "chemical_space_map.html"
        fig.write_html(output_path)
        
        return str(output_path)
    
    def create_sar_cliff_analysis(self, df: pd.DataFrame) -> str:
        """Identify and visualize SAR cliffs (similar structures, different activity)."""
        if not RDKIT_AVAILABLE or len(df) < 10:
            return None
        
        # Calculate pairwise similarities and activity differences
        cliff_data = []
        activity_col = 'ht2a_affinity_pred' if 'ht2a_affinity_pred' in df.columns else 'drug_likeness'
        
        for i, row1 in df.iterrows():
            mol1 = Chem.MolFromSmiles(row1['smiles'])
            if not mol1:
                continue
                
            fp1 = FingerprintMols.FingerprintMol(mol1)
            
            for j, row2 in df.iterrows():
                if i >= j:  # Avoid duplicates
                    continue
                    
                mol2 = Chem.MolFromSmiles(row2['smiles'])
                if not mol2:
                    continue
                
                fp2 = FingerprintMols.FingerprintMol(mol2)
                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                
                activity_diff = abs(row1[activity_col] - row2[activity_col])
                
                # SAR cliff criteria: high similarity, high activity difference
                if similarity > 0.7 and activity_diff > 1.0:
                    cliff_data.append({
                        'compound1': row1['name'],
                        'compound2': row2['name'],
                        'similarity': similarity,
                        'activity_diff': activity_diff,
                        'activity1': row1[activity_col],
                        'activity2': row2[activity_col]
                    })
        
        if not cliff_data:
            return None
        
        # Create cliff visualization
        cliff_df = pd.DataFrame(cliff_data)
        
        fig = px.scatter(
            cliff_df,
            x='similarity',
            y='activity_diff',
            hover_data=['compound1', 'compound2', 'activity1', 'activity2'],
            title='SAR Cliff Analysis<br>High Similarity vs High Activity Difference',
            labels={'similarity': 'Structural Similarity (Tanimoto)',
                   'activity_diff': f'{activity_col.replace("_", " ").title()} Difference'}
        )
        
        # Add threshold lines
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                     annotation_text="Activity Cliff Threshold")
        fig.add_vline(x=0.7, line_dash="dash", line_color="blue",
                     annotation_text="Similarity Threshold")
        
        fig.update_layout(width=800, height=600)
        
        # Save plot
        output_path = self.output_dir / "sar_cliff_analysis.html"
        fig.write_html(output_path)
        
        return str(output_path)
    
    def create_multi_parameter_dashboard(self, df: pd.DataFrame) -> str:
        """Create comprehensive multi-parameter SAR dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MW vs LogP', 'TPSA vs Activity', 'Drug-likeness Distribution', 'BBB Penetration'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: MW vs LogP colored by activity
        activity_col = 'ht2a_affinity_pred' if 'ht2a_affinity_pred' in df.columns else 'drug_likeness'
        
        fig.add_trace(
            go.Scatter(
                x=df['mw'] if 'mw' in df.columns else [0]*len(df),
                y=df['logp'] if 'logp' in df.columns else [0]*len(df),
                mode='markers',
                marker=dict(
                    color=df[activity_col] if activity_col in df.columns else [0]*len(df),
                    colorscale='Viridis',
                    size=8,
                    showscale=True,
                    colorbar=dict(x=0.45, title=activity_col.replace('_', ' ').title())
                ),
                text=df['name'],
                name='Compounds'
            ),
            row=1, col=1
        )
        
        # Plot 2: TPSA vs Activity
        if 'tpsa' in df.columns and activity_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['tpsa'],
                    y=df[activity_col],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    text=df['name'],
                    name='TPSA vs Activity'
                ),
                row=1, col=2
            )
        
        # Plot 3: Drug-likeness distribution
        if 'drug_likeness' in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df['drug_likeness'],
                    nbinsx=10,
                    name='Drug-likeness',
                    marker_color='skyblue'
                ),
                row=2, col=1
            )
        
        # Plot 4: BBB penetration
        if 'bbb_label' in df.columns:
            bbb_counts = df['bbb_label'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=bbb_counts.index,
                    values=bbb_counts.values,
                    name="BBB Penetration"
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Multi-Parameter SAR Dashboard",
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Molecular Weight (Da)", row=1, col=1)
        fig.update_yaxes(title_text="LogP", row=1, col=1)
        fig.update_xaxes(title_text="TPSA (Ų)", row=1, col=2)
        fig.update_yaxes(title_text=activity_col.replace('_', ' ').title(), row=1, col=2)
        fig.update_xaxes(title_text="Drug-likeness Score", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        # Save plot
        output_path = self.output_dir / "multi_parameter_dashboard.html"
        fig.write_html(output_path)
        
        return str(output_path)
    
    def generate_sar_report(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate comprehensive SAR analysis report."""
        print("🎨 Generating SAR visualizations...")
        
        report_files = {}
        
        # 1. Enhanced correlation heatmap
        try:
            heatmap_path = self.create_property_correlation_heatmap(df)
            if heatmap_path:
                report_files['correlation_heatmap'] = heatmap_path
                print("✅ Correlation heatmap created")
        except Exception as e:
            print(f"❌ Correlation heatmap failed: {str(e)}")
        
        # 2. Activity landscape
        try:
            landscape_path = self.create_activity_landscape(df)
            if landscape_path:
                report_files['activity_landscape'] = landscape_path
                print("✅ Activity landscape created")
        except Exception as e:
            print(f"❌ Activity landscape failed: {str(e)}")
        
        # 3. Chemical space map
        try:
            space_path = self.create_chemical_space_map(df)
            if space_path:
                report_files['chemical_space'] = space_path
                print("✅ Chemical space map created")
        except Exception as e:
            print(f"❌ Chemical space map failed: {str(e)}")
        
        # 4. SAR cliff analysis
        try:
            cliff_path = self.create_sar_cliff_analysis(df)
            if cliff_path:
                report_files['sar_cliffs'] = cliff_path
                print("✅ SAR cliff analysis created")
        except Exception as e:
            print(f"❌ SAR cliff analysis failed: {str(e)}")
        
        # 5. Multi-parameter dashboard
        try:
            dashboard_path = self.create_multi_parameter_dashboard(df)
            if dashboard_path:
                report_files['multi_dashboard'] = dashboard_path
                print("✅ Multi-parameter dashboard created")
        except Exception as e:
            print(f"❌ Multi-parameter dashboard failed: {str(e)}")
        
        return report_files

if __name__ == "__main__":
    # Test SAR visualizer
    print("🎨 Testing SAR Heatmaps and Visual Analytics...")
    
    visualizer = SARVisualizer()
    
    # Create test data
    test_data = {
        'name': ['2C-B', '2C-I', 'DOB', 'Mescaline'],
        'smiles': [
            'CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN',
            'CCc1cc(I)c(OCc2ccccc2)c(I)c1CCN', 
            'CC(N)Cc1cc(Br)c(OCc2ccccc2)c(Br)c1',
            'COc1cc(CCN)cc(OC)c1OC'
        ],
        'class': ['2C-series', '2C-series', 'DOx-series', 'Mescaline-analog'],
        'mw': [334.1, 428.1, 320.1, 211.3],
        'logp': [3.2, 3.8, 3.1, 0.4],
        'tpsa': [45.2, 45.2, 45.2, 58.9],
        'drug_likeness': [1.0, 0.75, 1.0, 1.0],
        'ht2a_affinity_pred': [8.7, 8.9, 8.2, 6.2]
    }
    
    test_df = pd.DataFrame(test_data)
    
    # Test correlation heatmap
    heatmap_path = visualizer.create_property_correlation_heatmap(test_df)
    if heatmap_path:
        print(f"✅ Test heatmap created: {heatmap_path}")
    
    print("✅ SAR Visualization Tools Ready!")