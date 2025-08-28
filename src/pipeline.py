"""
Main pipeline for psychedelic therapeutics analysis.
Orchestrates data loading, descriptor computation, visualization, and export.
"""

import pandas as pd
from pathlib import Path
import sys
import os

# Add src directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from data import load_demo, validate_smiles_column
from descriptors import compute, get_descriptor_stats
from viz2d import plot_dashboard, plot_correlation_heatmap, plot_class_comparison
from viz3d import batch_convert_to_sdf, setup_output_dir

def setup_directories():
    """Create necessary directories."""
    directories = ['outputs', 'data']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def print_summary(df: pd.DataFrame):
    """Print summary of the analysis."""
    print("\n" + "="*60)
    print("🧬 PSYCHEDELIC THERAPEUTICS ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   • Total compounds: {len(df)}")
    print(f"   • Compound classes: {df['class'].nunique()}")
    print(f"   • Classes: {', '.join(df['class'].unique())}")
    
    # BBB penetration summary
    if 'bbb_label' in df.columns:
        bbb_counts = df['bbb_label'].value_counts()
        print(f"\n🧠 Blood-Brain Barrier Penetration:")
        for label, count in bbb_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   • {label}: {count} compounds ({percentage:.1f}%)")
    
    # Drug-likeness summary
    if 'drug_likeness' in df.columns:
        avg_drug_likeness = df['drug_likeness'].mean()
        print(f"\n💊 Drug-likeness Assessment:")
        print(f"   • Average score: {avg_drug_likeness:.2f}/1.0")
        
        excellent = len(df[df['drug_likeness'] >= 0.8])
        good = len(df[(df['drug_likeness'] >= 0.6) & (df['drug_likeness'] < 0.8)])
        poor = len(df[df['drug_likeness'] < 0.6])
        
        print(f"   • Excellent (≥0.8): {excellent} compounds")
        print(f"   • Good (0.6-0.8): {good} compounds")
        print(f"   • Poor (<0.6): {poor} compounds")
    
    # Molecular property ranges
    if 'mw' in df.columns:
        print(f"\n⚗️ Molecular Properties:")
        print(f"   • Molecular weight: {df['mw'].min():.1f} - {df['mw'].max():.1f} Da")
        print(f"   • LogP: {df['logp'].min():.1f} - {df['logp'].max():.1f}")
        print(f"   • TPSA: {df['tpsa'].min():.1f} - {df['tpsa'].max():.1f} Ų")
    
    print(f"\n🎯 Key Findings:")
    
    # Best compounds by drug-likeness
    if 'drug_likeness' in df.columns:
        best_compounds = df.nlargest(3, 'drug_likeness')
        print(f"   • Top compounds by drug-likeness:")
        for _, row in best_compounds.iterrows():
            print(f"     - {row['name']}: {row['drug_likeness']:.2f}")
    
    # CNS-favorable compounds
    if 'bbb_label' in df.columns:
        cns_favorable = df[df['bbb_label'] == 'Good BBB']
        if len(cns_favorable) > 0:
            print(f"   • CNS-favorable compounds ({len(cns_favorable)}):")
            for _, row in cns_favorable.head(3).iterrows():
                print(f"     - {row['name']} (TPSA: {row['tpsa']:.1f} Ų)")

def run():
    """Run the complete analysis pipeline."""
    print("🧬 Starting Psychedelic Therapeutics Analysis Pipeline")
    print("="*60)
    
    try:
        # Setup
        print("\n1️⃣ Setting up directories...")
        setup_directories()
        
        # Load data
        print("\n2️⃣ Loading compound data...")
        df = load_demo()
        print(f"   ✅ Loaded {len(df)} compounds")
        
        # Validate SMILES
        print("\n3️⃣ Validating SMILES...")
        validate_smiles_column(df)
        print("   ✅ SMILES validation passed")
        
        # Compute descriptors
        print("\n4️⃣ Computing molecular descriptors...")
        df_with_descriptors = compute(df)
        print("   ✅ Descriptors computed successfully")
        
        # Show sample results
        print("\n   📋 Sample results:")
        sample_cols = ['name', 'class', 'mw', 'logp', 'tpsa', 'drug_likeness', 'bbb_label']
        available_cols = [col for col in sample_cols if col in df_with_descriptors.columns]
        print(df_with_descriptors[available_cols].head(3).to_string(index=False))
        
        # Generate 2D visualizations
        print("\n5️⃣ Generating 2D visualizations...")
        
        # Main dashboard
        dashboard_path = plot_dashboard(df_with_descriptors)
        print(f"   ✅ Dashboard: {dashboard_path}")
        
        # Correlation heatmap
        try:
            heatmap_path = plot_correlation_heatmap(df_with_descriptors)
            print(f"   ✅ Correlation heatmap: {heatmap_path}")
        except Exception as e:
            print(f"   ⚠️ Correlation heatmap failed: {str(e)}")
        
        # Class comparison
        try:
            comparison_path = plot_class_comparison(df_with_descriptors)
            print(f"   ✅ Class comparison: {comparison_path}")
        except Exception as e:
            print(f"   ⚠️ Class comparison failed: {str(e)}")
        
        # Generate 3D structures and SDF export
        print("\n6️⃣ Generating 3D structures...")
        try:
            sdf_path = batch_convert_to_sdf(df_with_descriptors)
            print(f"   ✅ SDF export: {sdf_path}")
        except Exception as e:
            print(f"   ❌ 3D structure generation failed: {str(e)}")
            print("   This may be due to RDKit installation issues")
        
        # Print comprehensive summary
        print_summary(df_with_descriptors)
        
        # List generated files
        print(f"\n📁 Generated Files:")
        output_dir = Path("outputs")
        if output_dir.exists():
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    print(f"   ✅ {file_path.name} ({file_size:,} bytes)")
        
        print(f"\n🚀 Next Steps:")
        print(f"   • Run Streamlit app: streamlit run app/streamlit_app.py")
        print(f"   • Open Jupyter notebook: jupyter lab")
        print(f"   • View interactive 3D: Open the Streamlit app")
        print(f"   • Explore generated files in outputs/ directory")
        
        print(f"\n🎉 Pipeline completed successfully!")
        print("="*60)
        
        return df_with_descriptors
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = run()
    if result is not None:
        print("Pipeline execution completed successfully!")
    else:
        print("Pipeline execution failed!")
        sys.exit(1)