"""
Streamlit web application for interactive psychedelic therapeutics analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import base64
from io import BytesIO
import time

# Add src directory to path
current_dir = Path(__file__).parent.parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from data import load_demo, load_custom_csv, validate_smiles_column
    from descriptors import compute, smiles_to_mol, calculate_descriptors, calculate_drug_likeness, bbb_label
    from viz2d import plot_dashboard, plot_correlation_heatmap
    from viz3d import smiles_to_3d_viewer, smiles_to_molblock, save_single_sdf, get_mol_image_base64
    from psychedelic_sar import TwoCBAnalyzer, batch_analyze_2cb_analogs
    from ht2a_binding import HT2AReceptorPredictor, analyze_5ht2a_binding_batch
    from novel_analog_generator import NovelAnalogGenerator, generate_psychedelic_library
    import py3Dmol
    from rdkit import Chem
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Psychedelic Therapeutics Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def setup_session_state():
    """Initialize session state variables."""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'selected_molecule' not in st.session_state:
        st.session_state.selected_molecule = None
    if 'custom_smiles' not in st.session_state:
        st.session_state.custom_smiles = ""

def load_data():
    """Load and process data."""
    data_option = st.sidebar.selectbox(
        "Select Data Source",
        ["Demo Dataset", "Upload CSV"]
    )
    
    df = None
    
    if data_option == "Demo Dataset":
        try:
            df = load_demo()
            st.sidebar.success(f"✅ Loaded {len(df)} demo compounds")
        except Exception as e:
            st.sidebar.error(f"Error loading demo data: {str(e)}")
            return None
    
    elif data_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV should have columns: class, name, smiles"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                validate_smiles_column(df)
                st.sidebar.success(f"✅ Loaded {len(df)} compounds from file")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
                return None
    
    if df is not None:
        # Compute descriptors if not already computed
        if 'mw' not in df.columns:
            with st.spinner("Computing molecular descriptors..."):
                df = compute(df)
        
        st.session_state.df = df
    
    return df

def display_dataset_overview(df):
    """Display dataset overview metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Compounds", len(df))
    
    with col2:
        st.metric("Compound Classes", df['class'].nunique())
    
    with col3:
        if 'drug_likeness' in df.columns:
            avg_drug_likeness = df['drug_likeness'].mean()
            st.metric("Avg Drug-likeness", f"{avg_drug_likeness:.2f}")
    
    with col4:
        if 'bbb_label' in df.columns:
            good_bbb = len(df[df['bbb_label'] == 'Good BBB'])
            st.metric("Good BBB", f"{good_bbb}/{len(df)}")

def molecule_selector(df):
    """Molecule selection widget."""
    st.subheader("🔍 Molecule Selection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Select from dataset
        molecule_names = df['name'].tolist()
        selected_name = st.selectbox(
            "Select molecule from dataset",
            molecule_names,
            key="molecule_selector"
        )
        
        if selected_name:
            selected_row = df[df['name'] == selected_name].iloc[0]
            st.session_state.selected_molecule = selected_row
            
            # Display molecule info
            st.write("**Selected Molecule:**")
            info_cols = ['name', 'class', 'smiles']
            if 'mw' in selected_row:
                info_cols.extend(['mw', 'logp', 'tpsa', 'drug_likeness', 'bbb_label'])
            
            for col in info_cols:
                if col in selected_row:
                    value = selected_row[col]
                    if isinstance(value, float):
                        st.write(f"**{col.upper()}:** {value:.2f}")
                    else:
                        st.write(f"**{col.upper()}:** {value}")
    
    with col2:
        # Custom SMILES input
        st.write("**Or enter custom SMILES:**")
        custom_smiles = st.text_input(
            "SMILES string",
            value=st.session_state.custom_smiles,
            placeholder="CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN",
            key="custom_smiles_input"
        )
        
        if custom_smiles and custom_smiles != st.session_state.custom_smiles:
            st.session_state.custom_smiles = custom_smiles
            
            # Validate and compute properties for custom SMILES
            mol = smiles_to_mol(custom_smiles)
            if mol is not None:
                descriptors = calculate_descriptors(mol)
                drug_likeness = calculate_drug_likeness(descriptors)
                bbb = bbb_label(descriptors['tpsa'])
                
                # Create custom molecule entry
                custom_molecule = {
                    'name': 'Custom Molecule',
                    'class': 'Custom',
                    'smiles': custom_smiles,
                    'mw': descriptors['mw'],
                    'logp': descriptors['logp'],
                    'tpsa': descriptors['tpsa'],
                    'hbd': descriptors['hbd'],
                    'hba': descriptors['hba'],
                    'rotb': descriptors['rotb'],
                    'rings': descriptors['rings'],
                    'drug_likeness': drug_likeness,
                    'bbb_label': bbb
                }
                
                st.session_state.selected_molecule = pd.Series(custom_molecule)
                
                # Display computed properties
                st.success("✅ Valid SMILES - Properties computed")
                st.write(f"**MW:** {descriptors['mw']:.1f} Da")
                st.write(f"**LogP:** {descriptors['logp']:.2f}")
                st.write(f"**TPSA:** {descriptors['tpsa']:.1f} Ų")
                st.write(f"**Drug-likeness:** {drug_likeness:.2f}")
                st.write(f"**BBB:** {bbb}")
            else:
                st.error("❌ Invalid SMILES string")
                st.session_state.selected_molecule = None

def display_3d_viewer(molecule):
    """Display 3D molecular viewer."""
    if molecule is None:
        st.info("Please select a molecule to view in 3D")
        return
    
    st.subheader(f"🧬 3D Structure: {molecule['name']}")
    
    try:
        # Generate 3D viewer
        smiles = molecule['smiles']
        
        with st.spinner("Generating 3D structure..."):
            view = smiles_to_3d_viewer(smiles, width=800, height=600)
        
        if view is not None:
            # Display 3D viewer
            viewer_html = view._make_html()
            st.components.v1.html(viewer_html, height=650)
            
            # Style options
            col1, col2 = st.columns(2)
            with col1:
                style = st.selectbox(
                    "Visualization Style",
                    ["stick", "sphere", "line"],
                    key="viz_style"
                )
            
            with col2:
                if st.button("🔄 Refresh 3D View"):
                    st.experimental_rerun()
            
        else:
            st.error("❌ Could not generate 3D structure for this molecule")
            
    except Exception as e:
        st.error(f"Error generating 3D view: {str(e)}")

def export_functions(molecule):
    """Export functionality for selected molecule."""
    if molecule is None:
        return
    
    st.subheader("💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export SDF", key="export_sdf"):
            try:
                with st.spinner("Generating SDF..."):
                    sdf_path = save_single_sdf(
                        molecule['smiles'], 
                        molecule['name'], 
                        f"{molecule['name']}_export.sdf"
                    )
                st.success(f"✅ SDF exported to: {sdf_path}")
            except Exception as e:
                st.error(f"❌ SDF export failed: {str(e)}")
    
    with col2:
        if st.button("🖼️ Export 2D PNG", key="export_2d"):
            try:
                with st.spinner("Generating 2D image..."):
                    img_b64 = get_mol_image_base64(molecule['smiles'])
                
                if img_b64:
                    # Create download link
                    img_data = base64.b64decode(img_b64)
                    st.download_button(
                        label="⬇️ Download 2D PNG",
                        data=img_data,
                        file_name=f"{molecule['name']}_2D.png",
                        mime="image/png"
                    )
                else:
                    st.error("❌ Could not generate 2D image")
            except Exception as e:
                st.error(f"❌ 2D image export failed: {str(e)}")
    
    with col3:
        if st.button("📊 Export Properties", key="export_props"):
            # Create properties DataFrame
            props_data = {
                'Property': ['Name', 'Class', 'SMILES', 'MW (Da)', 'LogP', 'TPSA (Ų)', 
                           'HBD', 'HBA', 'Rotatable Bonds', 'Rings', 'Drug-likeness', 'BBB Label'],
                'Value': [
                    molecule.get('name', 'N/A'),
                    molecule.get('class', 'N/A'),
                    molecule.get('smiles', 'N/A'),
                    f"{molecule.get('mw', 0):.1f}",
                    f"{molecule.get('logp', 0):.2f}",
                    f"{molecule.get('tpsa', 0):.1f}",
                    f"{molecule.get('hbd', 0):.0f}",
                    f"{molecule.get('hba', 0):.0f}",
                    f"{molecule.get('rotb', 0):.0f}",
                    f"{molecule.get('rings', 0):.0f}",
                    f"{molecule.get('drug_likeness', 0):.2f}",
                    molecule.get('bbb_label', 'N/A')
                ]
            }
            
            props_df = pd.DataFrame(props_data)
            csv_data = props_df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Download Properties CSV",
                data=csv_data,
                file_name=f"{molecule['name']}_properties.csv",
                mime="text/csv"
            )

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🧬 Psychedelic Therapeutics Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Interactive analysis of psychedelic compounds with 2D analytics and 3D molecular visualization**")
    
    # Initialize session state
    setup_session_state()
    
    # Sidebar
    st.sidebar.title("🔧 Control Panel")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Please load data to continue")
        st.stop()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🧬 3D Viewer", "📈 Analytics", "🎯 Advanced SAR", "🤖 AI Generation", "💾 Export"])
    
    with tab1:
        st.header("📊 Dataset Overview")
        display_dataset_overview(df)
        
        # Dataset table
        st.subheader("📋 Compound Data")
        display_cols = ['name', 'class', 'smiles']
        if 'mw' in df.columns:
            display_cols.extend(['mw', 'logp', 'tpsa', 'drug_likeness', 'bbb_label'])
        
        st.dataframe(df[display_cols], use_container_width=True)
        
        # Display dashboard image if available
        dashboard_path = Path("outputs/dashboard.png")
        if dashboard_path.exists():
            st.subheader("📈 Analysis Dashboard")
            st.image(str(dashboard_path), caption="Psychedelic Therapeutics Analysis Dashboard")
    
    with tab2:
        st.header("🧬 Interactive 3D Molecular Viewer")
        
        # Molecule selection
        molecule_selector(df)
        
        # 3D viewer
        if st.session_state.selected_molecule is not None:
            display_3d_viewer(st.session_state.selected_molecule)
        
        # Export options
        export_functions(st.session_state.selected_molecule)
    
    with tab3:
        st.header("📈 Advanced Analytics")
        
        if len(df) > 1:
            # Generate visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎨 Generate Dashboard"):
                    with st.spinner("Creating analysis dashboard..."):
                        try:
                            dashboard_path = plot_dashboard(df)
                            st.success(f"✅ Dashboard created: {dashboard_path}")
                            st.image(dashboard_path)
                        except Exception as e:
                            st.error(f"❌ Dashboard creation failed: {str(e)}")
            
            with col2:
                if st.button("🔥 Generate Heatmap"):
                    with st.spinner("Creating correlation heatmap..."):
                        try:
                            heatmap_path = plot_correlation_heatmap(df)
                            st.success(f"✅ Heatmap created: {heatmap_path}")
                            st.image(heatmap_path)
                        except Exception as e:
                            st.error(f"❌ Heatmap creation failed: {str(e)}")
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            if 'mw' in df.columns:
                stats_df = df[['mw', 'logp', 'tpsa', 'drug_likeness']].describe()
                st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("Load more compounds for advanced analytics")
    
    with tab4:
        st.header("🎯 Advanced SAR Analysis for CNS Therapeutics")
        st.markdown("**Specialized analysis for 5-HT2A receptor-targeted psychedelic drug design**")
        
        # Enhanced SAR Analysis Section
        st.subheader("🧬 2C-B Analog Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔬 Run Enhanced SAR Analysis", key="enhanced_sar"):
                with st.spinner("Running advanced SAR analysis..."):
                    try:
                        # Run batch analysis on all compounds
                        sar_results = batch_analyze_2cb_analogs(df)
                        st.session_state.sar_results = sar_results
                        st.success("✅ Enhanced SAR analysis completed!")
                    except Exception as e:
                        st.error(f"❌ SAR analysis failed: {str(e)}")
        
        with col2:
            if st.button("🧠 Run 5-HT2A Binding Prediction", key="ht2a_binding"):
                with st.spinner("Predicting 5-HT2A binding affinities..."):
                    try:
                        # Run 5-HT2A binding analysis
                        binding_results = analyze_5ht2a_binding_batch(df)
                        st.session_state.binding_results = binding_results
                        st.success("✅ 5-HT2A binding analysis completed!")
                    except Exception as e:
                        st.error(f"❌ Binding analysis failed: {str(e)}")
        
        # Display SAR Results
        if 'sar_results' in st.session_state:
            st.subheader("📊 Enhanced SAR Results")
            sar_df = st.session_state.sar_results
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_cns_mpo = sar_df['cns_mpo'].mean() if 'cns_mpo' in sar_df.columns else 0
                st.metric("Avg CNS MPO", f"{avg_cns_mpo:.2f}/6")
            with col2:
                avg_bbb = sar_df['bbb_score'].mean() if 'bbb_score' in sar_df.columns else 0
                st.metric("Avg BBB Score", f"{avg_bbb:.3f}")
            with col3:
                high_affinity = len(sar_df[sar_df['ht2a_affinity_pred'] > 7]) if 'ht2a_affinity_pred' in sar_df.columns else 0
                st.metric("High Affinity (>7)", high_affinity)
            with col4:
                drug_like = len(sar_df[sar_df['lipinski_violations'] == 0]) if 'lipinski_violations' in sar_df.columns else 0
                st.metric("Drug-like (Lipinski)", drug_like)
            
            # Display detailed results
            display_cols = ['name', 'cns_mpo', 'bbb_score', 'ht2a_affinity_pred', 'pharmacophore_score', 'synthesis_difficulty']
            available_cols = [col for col in display_cols if col in sar_df.columns]
            st.dataframe(sar_df[available_cols], use_container_width=True)
            
            # SAR Insights
            st.subheader("💡 SAR Insights")
            selected_compound = st.selectbox("Select compound for detailed analysis:", sar_df['name'].tolist())
            if selected_compound:
                compound_row = sar_df[sar_df['name'] == selected_compound].iloc[0]
                insights = compound_row.get('sar_insights', '').split(' | ')
                for insight in insights:
                    if insight.strip():
                        st.write(f"• {insight}")
        
        # Display 5-HT2A Binding Results
        if 'binding_results' in st.session_state:
            st.subheader("🧠 5-HT2A Binding Predictions")
            binding_df = st.session_state.binding_results
            
            # Binding metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_pki = binding_df['pki_predicted'].mean() if 'pki_predicted' in binding_df.columns else 0
                st.metric("Avg Predicted pKi", f"{avg_pki:.2f}")
            with col2:
                high_activity = len(binding_df[binding_df['activity_class'] == 'High Activity']) if 'activity_class' in binding_df.columns else 0
                st.metric("High Activity", high_activity)
            with col3:
                high_confidence = len(binding_df[binding_df['confidence'] > 0.8]) if 'confidence' in binding_df.columns else 0
                st.metric("High Confidence", high_confidence)
            
            # Display binding predictions
            binding_cols = ['name', 'pki_predicted', 'activity_class', 'confidence', 'overall_selectivity']
            available_binding_cols = [col for col in binding_cols if col in binding_df.columns]
            st.dataframe(binding_df[available_binding_cols], use_container_width=True)
            
            # Individual compound analysis
            st.subheader("🔍 Individual Compound Analysis")
            if st.session_state.selected_molecule is not None:
                mol_name = st.session_state.selected_molecule['name']
                mol_smiles = st.session_state.selected_molecule['smiles']
                
                # Run individual analysis
                try:
                    analyzer = TwoCBAnalyzer()
                    individual_analysis = analyzer.analyze_2cb_analog(mol_smiles, mol_name)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Structural Classification:**")
                        st.write(f"Scaffold: {individual_analysis['structural_classification']['scaffold']}")
                        st.write(f"Pattern: {individual_analysis['structural_classification']['substitution_pattern']}")
                        
                        st.write("**Optimization Suggestions:**")
                        for suggestion in individual_analysis['optimization_suggestions']:
                            st.write(f"• {suggestion}")
                    
                    with col2:
                        st.write("**SAR Insights:**")
                        for insight in individual_analysis['sar_insights']:
                            st.write(f"• {insight}")
                        
                        if individual_analysis['safety_flags']:
                            st.write("**Safety Concerns:**")
                            for concern in individual_analysis['safety_flags']:
                                st.warning(f"⚠️ {concern}")
                        
                except Exception as e:
                    st.error(f"Error in individual analysis: {str(e)}")
    
    with tab5:
        st.header("🤖 AI-Powered Novel Analog Generation")
        st.markdown("**Generate novel 2C-B derivatives with optimized properties using AI-guided molecular design**")
        
        # Generation Parameters
        st.subheader("⚙️ Generation Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_compounds = st.slider(
                "Number of Analogs to Generate",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="More compounds = more diverse but slower generation"
            )
        
        with col2:
            target_profile = st.selectbox(
                "Optimization Target",
                [
                    "balanced_profile",
                    "high_potency", 
                    "selective_binding",
                    "oral_bioavailability"
                ],
                help="Select the primary optimization objective"
            )
        
        with col3:
            include_experimental = st.checkbox(
                "Include Experimental Scaffolds",
                value=True,
                help="Generate more diverse structures beyond traditional 2C scaffolds"
            )
        
        # Target Profile Information
        profile_descriptions = {
            "balanced_profile": "Optimizes for balanced 5-HT2A affinity, CNS penetration, and safety",
            "high_potency": "Maximizes predicted 5-HT2A binding affinity",
            "selective_binding": "Focuses on receptor selectivity and reduced off-targets",
            "oral_bioavailability": "Emphasizes drug-likeness and metabolic stability"
        }
        
        st.info(f"📊 **{target_profile}**: {profile_descriptions[target_profile]}")
        
        # Generation Controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧬 Generate Novel Analogs", type="primary", key="generate_analogs"):
                with st.spinner(f"Generating {num_compounds} novel 2C-B analogs..."):
                    try:
                        # Initialize generator
                        generator = NovelAnalogGenerator()
                        
                        # Generate analogs
                        generated_df = generator.generate_2cb_analogs(
                            num_analogs=num_compounds,
                            target_profile=target_profile
                        )
                        
                        # Store in session state
                        st.session_state.generated_compounds = generated_df
                        
                        st.success(f"✅ Successfully generated {len(generated_df)} novel analogs!")
                        
                    except Exception as e:
                        st.error(f"❌ Generation failed: {str(e)}")
        
        with col2:
            if st.button("📊 Generate Optimization Report", key="gen_report"):
                if 'generated_compounds' in st.session_state:
                    with st.spinner("Creating optimization report..."):
                        try:
                            generator = NovelAnalogGenerator()
                            report_path = generator.generate_optimization_report(
                                st.session_state.generated_compounds
                            )
                            
                            # Read and display report
                            with open(report_path, 'r') as f:
                                report_content = f.read()
                            
                            st.markdown(report_content)
                            
                            # Download button for report
                            st.download_button(
                                label="⬇️ Download Report",
                                data=report_content,
                                file_name="novel_analogs_report.md",
                                mime="text/markdown"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Report generation failed: {str(e)}")
                else:
                    st.warning("Please generate compounds first")
        
        # Display Generated Compounds
        if 'generated_compounds' in st.session_state:
            generated_df = st.session_state.generated_compounds
            
            st.subheader(f"🧬 Generated Compounds ({len(generated_df)} total)")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'optimization_score' in generated_df.columns:
                    avg_score = generated_df['optimization_score'].mean()
                    st.metric("Avg Optimization Score", f"{avg_score:.3f}")
            
            with col2:
                if 'ht2a_affinity' in generated_df.columns:
                    high_affinity = len(generated_df[generated_df['ht2a_affinity'] > 7])
                    st.metric("High 5-HT2A Affinity", f"{high_affinity}/{len(generated_df)}")
            
            with col3:
                if 'drug_likeness' in generated_df.columns:
                    drug_like = len(generated_df[generated_df['drug_likeness'] > 0.8])
                    st.metric("Drug-like Compounds", f"{drug_like}/{len(generated_df)}")
            
            with col4:
                if 'novelty_score' in generated_df.columns:
                    novel = len(generated_df[generated_df['novelty_score'] > 0.7])
                    st.metric("Novel Structures", f"{novel}/{len(generated_df)}")
            
            # Top compounds table
            st.subheader("🏆 Top 10 Compounds by Optimization Score")
            
            display_cols = ['name', 'smiles', 'modification_type']
            if 'optimization_score' in generated_df.columns:
                display_cols.append('optimization_score')
            if 'ht2a_affinity' in generated_df.columns:
                display_cols.append('ht2a_affinity')
            if 'bbb_score' in generated_df.columns:
                display_cols.append('bbb_score')
            if 'drug_likeness' in generated_df.columns:
                display_cols.append('drug_likeness')
            
            available_cols = [col for col in display_cols if col in generated_df.columns]
            
            # Sort by optimization score if available
            if 'optimization_score' in generated_df.columns:
                top_compounds = generated_df.nlargest(10, 'optimization_score')
            else:
                top_compounds = generated_df.head(10)
            
            st.dataframe(top_compounds[available_cols], use_container_width=True)
            
            # Individual compound analysis
            st.subheader("🔍 Detailed Compound Analysis")
            
            selected_generated = st.selectbox(
                "Select a generated compound for detailed analysis:",
                generated_df['name'].tolist(),
                key="generated_compound_selector"
            )
            
            if selected_generated:
                compound_row = generated_df[generated_df['name'] == selected_generated].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Compound Information:**")
                    st.write(f"**Name:** {compound_row['name']}")
                    st.write(f"**SMILES:** {compound_row['smiles']}")
                    st.write(f"**Modification Type:** {compound_row.get('modification_type', 'N/A')}")
                    st.write(f"**Description:** {compound_row.get('description', 'N/A')}")
                    
                    if 'optimization_score' in compound_row:
                        st.write(f"**Optimization Score:** {compound_row['optimization_score']:.3f}")
                
                with col2:
                    # Display 3D structure if possible
                    try:
                        selected_mol_dict = {
                            'name': compound_row['name'],
                            'smiles': compound_row['smiles'],
                            'class': 'Generated'
                        }
                        display_3d_viewer(selected_mol_dict)
                    except Exception as e:
                        st.error(f"Could not display 3D structure: {str(e)}")
                
                # Properties comparison with 2C-B
                st.write("**Property Comparison with 2C-B Reference:**")
                
                reference_props = {
                    'MW': 334.1, 'LogP': 3.2, 'TPSA': 45.2,
                    '5-HT2A Affinity': 8.7, 'BBB Score': 0.85
                }
                
                comparison_data = []
                for prop, ref_val in reference_props.items():
                    prop_key = prop.lower().replace('-', '_').replace(' ', '_')
                    
                    if prop_key in compound_row:
                        comp_val = compound_row[prop_key]
                        diff = comp_val - ref_val
                        percent_diff = (diff / ref_val) * 100 if ref_val != 0 else 0
                        
                        comparison_data.append({
                            'Property': prop,
                            '2C-B Reference': f"{ref_val:.2f}",
                            'Generated Compound': f"{comp_val:.2f}",
                            'Difference': f"{diff:+.2f}",
                            'Percent Change': f"{percent_diff:+.1f}%"
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
            
            # Export generated compounds
            st.subheader("💾 Export Generated Compounds")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV export
                csv_data = generated_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"generated_analogs_{target_profile}_{num_compounds}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # SDF export
                if st.button("🧬 Export to SDF", key="export_generated_sdf"):
                    try:
                        with st.spinner("Creating SDF file..."):
                            generator = NovelAnalogGenerator()
                            sdf_path = generator.export_compounds(
                                generated_df, 
                                f"generated_analogs_{target_profile}_{num_compounds}.csv"
                            )
                            st.success(f"✅ SDF exported: {sdf_path}")
                    except Exception as e:
                        st.error(f"❌ SDF export failed: {str(e)}")
            
            with col3:
                # Add to main dataset
                if st.button("➕ Add to Main Dataset", key="add_to_main"):
                    try:
                        # Select top 5 compounds to add
                        if 'optimization_score' in generated_df.columns:
                            top_5 = generated_df.nlargest(5, 'optimization_score')
                        else:
                            top_5 = generated_df.head(5)
                        
                        # Add to main dataframe
                        enhanced_main_df = pd.concat([df, top_5], ignore_index=True)
                        st.session_state.df = enhanced_main_df
                        
                        st.success(f"✅ Added top {len(top_5)} compounds to main dataset")
                        st.info("Switch to other tabs to analyze the enhanced dataset")
                        
                    except Exception as e:
                        st.error(f"❌ Failed to add compounds: {str(e)}")
        
        else:
            st.info("💡 Click 'Generate Novel Analogs' to start creating new 2C-B derivatives")
    
    with tab6:
        st.header("💾 Bulk Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export All to SDF"):
                try:
                    with st.spinner("Converting all molecules to 3D..."):
                        from viz3d import batch_convert_to_sdf
                        sdf_path = batch_convert_to_sdf(df, output_file="all_compounds.sdf")
                    st.success(f"✅ All compounds exported to: {sdf_path}")
                except Exception as e:
                    st.error(f"❌ Bulk SDF export failed: {str(e)}")
        
        with col2:
            # Export full dataset as CSV
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Full Dataset CSV",
                data=csv_data,
                file_name="psychedelic_compounds_full.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("**🧬 Psychedelic Therapeutics Dashboard** | Built with Streamlit, RDKit, and py3Dmol")

if __name__ == "__main__":
    main()