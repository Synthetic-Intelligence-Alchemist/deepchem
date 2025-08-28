# 🧬 Psychedelic Therapeutics Analysis Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/RDKit-2022.9.5+-green.svg)](https://www.rdkit.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0+-red.svg)](https://streamlit.io/)

**A comprehensive computational platform for analyzing psychedelic compounds with both 2D analytics and interactive 3D molecular visualization.**

## 🎯 Features

- **📊 2D Analytics Dashboard**: 4-panel visualization with molecular property distributions
- **🧬 Interactive 3D Viewer**: py3Dmol-powered molecular visualization
- **⚗️ Molecular Descriptors**: MW, LogP, TPSA, HBD, HBA, rotatable bonds, drug-likeness
- **🧠 CNS Predictions**: Blood-brain barrier penetration assessment
- **📁 Export Capabilities**: SDF, PNG, and CSV export functionality
- **🔬 Ready-to-Run**: Works in VS Code, Qoder, and Jupyter environments

## 📁 Repository Structure

```
psychedelic-therapeutics/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   └── compounds_demo.csv            # Demo dataset (12 psychedelic compounds)
├── src/
│   ├── data.py                       # Data loading utilities
│   ├── descriptors.py                # Molecular descriptor calculations
│   ├── viz2d.py                      # 2D visualization dashboard
│   ├── viz3d.py                      # 3D molecular visualization
│   └── pipeline.py                   # Main orchestration pipeline
├── app/
│   └── streamlit_app.py              # Interactive web interface
├── notebooks/
│   └── psychedelic_dashboard.ipynb   # Jupyter analysis notebook
├── tests/
│   └── test_smiles.py                # Test suite for SMILES parsing
└── outputs/                          # Generated files (created automatically)
    ├── dashboard.png                 # 2D analysis dashboard
    ├── compounds.sdf                 # 3D molecular structures
    └── *.png, *.sdf, *.csv           # Exported files
```

## 🚀 Quick Start

### 1. Environment Setup

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python -m src.pipeline
```

### 3. Launch Interactive Apps

**Streamlit Web App:**
```bash
streamlit run app/streamlit_app.py
```

**Jupyter Notebook:**
```bash
jupyter lab
# Open notebooks/psychedelic_dashboard.ipynb
```

## 📊 Demo Dataset

The platform includes 12 representative psychedelic compounds:

| Class | Compounds | Key Features |
|-------|-----------|--------------|
| **2C-series** | 2C-B, 2C-I, 2C-E, 2C-P, 2C-T-2 | Phenethylamine core, halogen substitutions |
| **DOx-series** | DOB, DOI, DOM | Amphetamine analogs of 2C compounds |
| **Mescaline-analog** | Mescaline, Escaline | Classic psychedelic alkaloids |
| **NBOMe-series** | 25B-NBOMe, 25I-NBOMe | Potent 5-HT2A agonists |

## 🧪 Usage Examples

### Command Line Analysis
```bash
# Run complete pipeline
python -m src.pipeline

# Run tests
python -m tests.test_smiles
```

### Python API
```python
from src.data import load_demo
from src.descriptors import compute
from src.viz2d import plot_dashboard
from src.viz3d import smiles_to_3d_viewer

# Load data and compute descriptors
df = load_demo()
df_with_descriptors = compute(df)

# Create 2D dashboard
plot_dashboard(df_with_descriptors)

# Create 3D viewer for 2C-B
smiles_2cb = "CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN"
viewer = smiles_to_3d_viewer(smiles_2cb)
viewer.show()
```

### Streamlit Interface

The Streamlit app provides:
- **📊 Dataset Overview**: Summary statistics and compound browser
- **🧬 3D Viewer**: Interactive molecular visualization with style options
- **📈 Analytics**: Generate dashboards and correlation heatmaps  
- **💾 Export**: Download SDF files, 2D images, and property data

## 🔬 Technical Details

### Molecular Descriptors Computed

| Property | Description | Importance |
|----------|-------------|------------|
| **MW** | Molecular Weight (Da) | Drug-likeness, bioavailability |
| **LogP** | Lipophilicity | Membrane permeability, BBB |
| **TPSA** | Topological Polar Surface Area (Ų) | BBB penetration |
| **HBD/HBA** | H-bond donors/acceptors | Binding affinity |
| **RotB** | Rotatable bonds | Flexibility, binding |
| **Rings** | Ring count | Structural complexity |
| **Drug-likeness** | Lipinski's Rule compliance (0-1) | Development potential |
| **BBB Label** | Brain penetration prediction | CNS activity |

### 3D Structure Generation

- **Algorithm**: RDKit ETKDG (Distance Geometry)
- **Optimization**: UFF (Universal Force Field)
- **Output**: MOL blocks compatible with visualization tools
- **Fallback**: Multiple embedding strategies for difficult molecules

### Visualization Capabilities

**2D Dashboard (4 panels):**
1. **Chemical Space**: MW vs LogP scatter plot by compound class
2. **TPSA Distribution**: Histogram with BBB threshold (60 Ų)
3. **Drug-likeness**: Average scores by compound class
4. **BBB Penetration**: Pie chart of penetration predictions

**3D Viewer Options:**
- **Stick**: Bond representation (default)
- **Sphere**: Space-filling model
- **Line**: Wireframe representation
- **Customizable**: Colors, background, size

## 📈 Analysis Results

### Sample Output for 2C-B:
```
🎯 2C-B Analysis:
   • Molecular Weight: 334.1 Da
   • LogP: 3.2 (optimal for BBB)
   • TPSA: 45.2 Ų (excellent CNS access)
   • Drug-likeness: 0.95/1.0 ✅
   • BBB Penetration: Good BBB ✅
   • Lipinski Violations: 0/4 ✅
```

### Generated Files:
- `outputs/dashboard.png` - 4-panel analysis dashboard
- `outputs/compounds.sdf` - All molecules in 3D SDF format
- `outputs/correlation_heatmap.png` - Property correlation matrix
- `outputs/class_comparison.png` - Inter-class property comparison

## 🔧 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'pandas'**
```bash
# Reinstall requirements
pip install --upgrade pip
pip install -r requirements.txt
```

**2. RDKit Installation Issues (Apple Silicon)**
```bash
# Use specific version
pip install rdkit-pypi==2022.9.5
# Or use conda
conda install -c conda-forge rdkit
```

**3. 3D Structure Generation Fails**
```bash
# Check RDKit installation
python -c "from rdkit import Chem; print('RDKit OK')"
# Verify SMILES validity
python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO'))"
```

**4. Streamlit App Won't Launch**
```bash
# Check Streamlit installation
streamlit --version
# Run with explicit Python
python -m streamlit run app/streamlit_app.py
```

**5. Jupyter Notebook Issues**
```bash
# Install Jupyter kernel
python -m ipykernel install --user --name psychedelic-env
# Launch with specific kernel
jupyter lab --no-browser
```

### Performance Tips

- **Large Datasets**: Process in batches of 100-500 molecules
- **3D Generation**: Disable optimization for faster processing
- **Memory**: Close Jupyter kernels when not in use
- **Browser**: Use Chrome/Firefox for best 3D visualization

### Dependencies Compatibility

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10+ | Required for latest features |
| RDKit | 2022.9.5+ | Core cheminformatics |
| py3Dmol | 2.0.0+ | 3D visualization |
| Streamlit | 1.25.0+ | Web interface |
| Pandas | 1.5.0+ | Data manipulation |
| Matplotlib | 3.5.0+ | 2D plotting |

## 🧬 Applications

### Research Areas
- **Psychedelic Medicine**: Therapeutic compound design
- **CNS Drug Discovery**: Blood-brain barrier optimization
- **Medicinal Chemistry**: Structure-activity relationships
- **Cheminformatics**: Molecular property prediction

### Educational Use
- **Computational Chemistry**: Hands-on molecular modeling
- **Drug Design**: Property-based optimization
- **Data Science**: Chemical data analysis workflows
- **Visualization**: 3D molecular representation

## 📚 Scientific Background

### Psychedelic Compound Classes

**2C-Series (Phenethylamines):**
- Core structure: 2,5-dimethoxyphenethylamine
- Substitution patterns affect potency and selectivity
- Primary target: 5-HT2A receptor

**DOx-Series (Amphetamines):** 
- Alpha-methylated 2C analogs
- Longer duration of action
- Higher potency than 2C counterparts

**Mescaline Analogs:**
- Classical psychedelic alkaloids
- 3,4,5-trimethoxyphenethylamine core
- Historical and cultural significance

### Drug Design Principles

**CNS Penetration:**
- TPSA < 60 Ų for good BBB penetration
- LogP 1-3 for optimal permeability
- MW < 450 Da for passive transport

**5-HT2A Receptor Targeting:**
- Phenethylamine/indole scaffolds
- Aromatic substitutions modulate activity
- Stereochemistry affects receptor selectivity

## 🤝 Contributing

We welcome contributions! Areas for enhancement:

- **New Descriptors**: Additional molecular properties
- **Advanced Visualizations**: Enhanced 2D/3D representations  
- **Dataset Expansion**: More psychedelic compound classes
- **Analysis Tools**: QSAR modeling, clustering algorithms
- **Export Formats**: MOL2, PDB, XYZ support

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- **RDKit Team**: Open-source cheminformatics toolkit
- **py3Dmol**: Browser-based 3D molecular visualization
- **Streamlit**: Rapid web app development
- **DeepChem Community**: Inspiration and molecular ML tools

## 📧 Support

For questions, issues, or collaboration opportunities:

- **Technical Issues**: Open a GitHub issue
- **Feature Requests**: Submit enhancement proposals
- **Research Collaboration**: Contact maintainers
- **Educational Use**: Documentation and tutorials available

---

**🧬 Advancing psychedelic therapeutics through computational chemistry** 🚀

*Built with ❤️ for the open-source scientific community*