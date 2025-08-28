"""
Analysis and Visualization Module
================================

Comprehensive analysis and visualization tools for psychedelic therapeutic design,
including SAR analysis, compound profiling, and interactive dashboards.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import base64
from io import BytesIO

class PsychedelicAnalyzer:
    """Comprehensive analyzer for psychedelic compounds."""
    
    def __init__(self):
        self.compounds_df = None
        self.models = {}
        self.color_palette = px.colors.qualitative.Set3
    
    def load_compound_data(self, df: pd.DataFrame):
        """Load compound dataset for analysis."""
        self.compounds_df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['smiles', 'compound_name']
        for col in required_cols:
            if col not in self.compounds_df.columns:
                print(f"Warning: Required column '{col}' not found")
    
    def create_compound_overview(self) -> Dict[str, any]:
        """Create comprehensive overview of compound dataset."""
        if self.compounds_df is None:
            return {}
        
        overview = {
            'total_compounds': len(self.compounds_df),
            'unique_compounds': self.compounds_df['smiles'].nunique(),
            'compound_classes': self.compounds_df['compound_class'].value_counts().to_dict() if 'compound_class' in self.compounds_df.columns else {},
            'property_statistics': {}
        }
        
        # Calculate property statistics
        numeric_cols = self.compounds_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            overview['property_statistics'][col] = {
                'mean': self.compounds_df[col].mean(),
                'std': self.compounds_df[col].std(),
                'min': self.compounds_df[col].min(),
                'max': self.compounds_df[col].max(),
                'median': self.compounds_df[col].median()
            }
        
        return overview
    
    def plot_property_distributions(self, properties: List[str] = None, 
                                  save_path: Optional[str] = None) -> go.Figure:
        """Plot distributions of molecular properties."""
        if self.compounds_df is None:
            return go.Figure()
        
        if properties is None:
            # Default properties to plot
            properties = ['mw', 'logp', 'tpsa', 'hbd', 'hba']
            properties = [p for p in properties if p in self.compounds_df.columns]
        
        n_props = len(properties)
        if n_props == 0:
            return go.Figure()
        
        # Create subplots
        cols = min(3, n_props)
        rows = (n_props + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=properties,
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
        )
        
        for i, prop in enumerate(properties):
            row = i // cols + 1
            col = i % cols + 1
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=self.compounds_df[prop],
                    name=prop,
                    opacity=0.7,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Molecular Property Distributions",
            height=300 * rows,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_chemical_space(self, x_prop: str = 'mw', y_prop: str = 'logp',
                           color_by: str = 'compound_class',
                           save_path: Optional[str] = None) -> go.Figure:
        """Plot chemical space visualization."""
        if self.compounds_df is None:
            return go.Figure()
        
        fig = px.scatter(
            self.compounds_df,
            x=x_prop,
            y=y_prop,
            color=color_by if color_by in self.compounds_df.columns else None,
            hover_data=['compound_name'] if 'compound_name' in self.compounds_df.columns else None,
            title=f"Chemical Space: {y_prop} vs {x_prop}",
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title=x_prop.upper(),
            yaxis_title=y_prop.upper(),
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_sar_heatmap(self, activity_col: str = 'pchembl_value',
                        property_cols: List[str] = None,
                        save_path: Optional[str] = None) -> go.Figure:
        """Create SAR heatmap showing correlations."""
        if self.compounds_df is None:
            return go.Figure()
        
        if activity_col not in self.compounds_df.columns:
            print(f"Activity column '{activity_col}' not found")
            return go.Figure()
        
        if property_cols is None:
            property_cols = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotatable_bonds']
            property_cols = [p for p in property_cols if p in self.compounds_df.columns]
        
        # Calculate correlations
        correlation_data = []
        for prop in property_cols:
            corr = self.compounds_df[prop].corr(self.compounds_df[activity_col])
            correlation_data.append({'Property': prop, 'Correlation': corr})
        
        corr_df = pd.DataFrame(correlation_data)
        
        fig = px.bar(
            corr_df,
            x='Property',
            y='Correlation',
            title=f"Property Correlations with {activity_col}",
            color='Correlation',
            color_continuous_scale='RdBu_r'
        )
        
        fig.update_layout(height=500)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def analyze_2cb_analogs(self, analog_smiles: List[str]) -> pd.DataFrame:
        """Analyze 2C-B analogs and compare properties."""
        # Reference 2C-B
        cb_smiles = 'CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN'
        
        results = []
        
        # Analyze 2C-B
        cb_mol = Chem.MolFromSmiles(cb_smiles)
        if cb_mol:
            cb_props = self._calculate_all_properties(cb_mol)
            cb_props.update({
                'compound_name': '2C-B',
                'smiles': cb_smiles,
                'is_reference': True
            })
            results.append(cb_props)
        
        # Analyze analogs
        for i, smiles in enumerate(analog_smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                props = self._calculate_all_properties(mol)
                props.update({
                    'compound_name': f'Analog_{i+1}',
                    'smiles': smiles,
                    'is_reference': False
                })
                results.append(props)
        
        df = pd.DataFrame(results)
        
        # Calculate differences from 2C-B
        if len(results) > 1:
            cb_props = results[0]
            for i in range(1, len(results)):
                for prop in ['mw', 'logp', 'tpsa', 'hbd', 'hba']:
                    if prop in cb_props and prop in results[i]:
                        diff_col = f'{prop}_diff_from_2cb'
                        df.loc[i, diff_col] = results[i][prop] - cb_props[prop]
        
        return df
    
    def _calculate_all_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate comprehensive molecular properties."""
        props = {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol),
            'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
        }
        
        # CNS-specific properties
        props.update({
            'bbb_score': self._estimate_bbb_penetration(props),
            'drug_likeness': self._calculate_drug_likeness(props),
        })
        
        return props
    
    def _estimate_bbb_penetration(self, props: Dict[str, float]) -> float:
        """Estimate blood-brain barrier penetration score."""
        # Simple BBB penetration model
        score = 1.0
        
        # Molecular weight penalty
        if props['mw'] > 450:
            score *= 0.5
        elif props['mw'] > 350:
            score *= 0.8
        
        # LogP score
        if 1.0 <= props['logp'] <= 3.0:
            score *= 1.0
        else:
            score *= 0.7
        
        # TPSA penalty
        if props['tpsa'] > 90:
            score *= 0.5
        elif props['tpsa'] > 60:
            score *= 0.8
        
        return score
    
    def _calculate_drug_likeness(self, props: Dict[str, float]) -> float:
        """Calculate drug-likeness score."""
        # Simplified drug-likeness based on Lipinski's rule
        violations = 0
        
        if props['mw'] > 500:
            violations += 1
        if props['logp'] > 5:
            violations += 1
        if props['hbd'] > 5:
            violations += 1
        if props['hba'] > 10:
            violations += 1
        
        return 1.0 - (violations / 4.0)
    
    def create_compound_dashboard(self, save_path: str = "psychedelic_dashboard.html"):
        """Create interactive dashboard for compound analysis."""
        if self.compounds_df is None:
            print("No compound data loaded")
            return
        
        # Create dashboard with multiple plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Molecular Weight vs LogP",
                "Property Distributions", 
                "Compound Classes",
                "Drug-likeness Score"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "pie"}, {"type": "bar"}]
            ]
        )
        
        # Plot 1: MW vs LogP
        if 'mw' in self.compounds_df.columns and 'logp' in self.compounds_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.compounds_df['mw'],
                    y=self.compounds_df['logp'],
                    mode='markers',
                    name='Compounds',
                    text=self.compounds_df.get('compound_name', ''),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Plot 2: Property distribution (TPSA)
        if 'tpsa' in self.compounds_df.columns:
            fig.add_trace(
                go.Histogram(
                    x=self.compounds_df['tpsa'],
                    name='TPSA',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Plot 3: Compound classes
        if 'compound_class' in self.compounds_df.columns:
            class_counts = self.compounds_df['compound_class'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=class_counts.index,
                    values=class_counts.values,
                    name="Classes",
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Plot 4: Drug-likeness scores
        if 'drug_likeness' in self.compounds_df.columns:
            fig.add_trace(
                go.Bar(
                    x=self.compounds_df['compound_name'][:10],  # Limit to first 10
                    y=self.compounds_df['drug_likeness'][:10],
                    name='Drug-likeness',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Psychedelic Therapeutics Analysis Dashboard",
            height=800
        )
        
        fig.write_html(save_path)
        print(f"Dashboard saved to {save_path}")

class MoleculeVisualizer:
    """Visualizer for molecular structures."""
    
    def __init__(self):
        self.img_size = (400, 400)
    
    def draw_molecule(self, smiles: str, title: str = "") -> str:
        """Draw molecule and return as base64 encoded image."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        
        # Generate 2D coordinates
        rdDepictor.Compute2DCoords(mol)
        
        # Draw molecule
        drawer = rdMolDraw2D.MolDraw2DCairo(self.img_size[0], self.img_size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Get image data
        img_data = drawer.GetDrawingText()
        
        # Encode as base64
        img_b64 = base64.b64encode(img_data).decode()
        
        return img_b64
    
    def create_molecule_grid(self, smiles_list: List[str], 
                           names: List[str] = None,
                           mols_per_row: int = 4) -> go.Figure:
        """Create grid of molecule images."""
        if names is None:
            names = [f"Molecule {i+1}" for i in range(len(smiles_list))]
        
        n_mols = len(smiles_list)
        rows = (n_mols + mols_per_row - 1) // mols_per_row
        
        fig = make_subplots(
            rows=rows, cols=mols_per_row,
            subplot_titles=names[:n_mols]
        )
        
        for i, smiles in enumerate(smiles_list):
            row = i // mols_per_row + 1
            col = i % mols_per_row + 1
            
            # Create molecule image
            img_b64 = self.draw_molecule(smiles, names[i] if i < len(names) else "")
            
            if img_b64:
                # Add as image
                fig.add_layout_image(
                    dict(
                        source=f"data:image/png;base64,{img_b64}",
                        xref="x domain", yref="y domain",
                        x=0, y=1, sizex=1, sizey=1,
                        xanchor="left", yanchor="top"
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Molecular Structures",
            height=200 * rows
        )
        
        # Hide axes
        for i in range(1, rows + 1):
            for j in range(1, mols_per_row + 1):
                fig.update_xaxes(visible=False, row=i, col=j)
                fig.update_yaxes(visible=False, row=i, col=j)
        
        return fig

def create_psychedelic_analyzer() -> PsychedelicAnalyzer:
    """Factory function to create psychedelic analyzer."""
    return PsychedelicAnalyzer()

def create_molecule_visualizer() -> MoleculeVisualizer:
    """Factory function to create molecule visualizer."""
    return MoleculeVisualizer()

if __name__ == "__main__":
    # Test analyzer
    analyzer = create_psychedelic_analyzer()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'smiles': [
            'CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN',  # 2C-B
            'CCc1cc(I)c(OCc2ccccc2)c(I)c1CCN',     # 2C-I
            'COc1cc(CCN)cc(OC)c1OC'                 # Mescaline
        ],
        'compound_name': ['2C-B', '2C-I', 'Mescaline'],
        'compound_class': ['2C-series', '2C-series', 'Mescaline-analog']
    })
    
    analyzer.load_compound_data(sample_data)
    overview = analyzer.create_compound_overview()
    print(f"Loaded {overview['total_compounds']} compounds")
    
    # Test visualizer
    visualizer = create_molecule_visualizer()
    print("Molecule visualizer created successfully")