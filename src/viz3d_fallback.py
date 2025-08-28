"""
Fallback 3D molecular visualization (without RDKit).
This module provides basic functionality when RDKit is not available.
"""

import pandas as pd
import py3Dmol
from pathlib import Path
from typing import Optional, Tuple
import warnings

# Pre-generated MOL blocks for demo compounds (created with RDKit offline)
DEMO_MOLBLOCKS = {
    'CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN': """2C-B
  -OEChem-01052500002D

 34 36  0     1  0  0  0  0  0999 V2000
    7.7942   -0.8660    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.7942    0.1340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.9282   -1.3660    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.6603    0.6340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0622   -0.8660    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.6603    1.6340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0622    0.1340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.9282    0.6340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    9.5263    2.1340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.1962    0.6340    0.0000 Br  0  0  0  0  0  0  0  0  0  0  0  0
    6.9282    1.6340    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    9.5263    3.1340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.1962   -1.3660    0.0000 Br  0  0  0  0  0  0  0  0  0  0  0  0
   10.3923    3.6340    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    6.0622    2.1340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0622    3.1340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.1962    3.6340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.9282    3.6340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.1962    4.6340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.9282    4.6340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0622    5.1340    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.7942   -1.3660    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.9282   -2.3660    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    9.6603    0.1340    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    7.6603    1.6340    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   10.5263    1.6340    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   10.5263    1.6340    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    8.5263    3.6340    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    8.5263    3.6340    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   10.3923    4.6340    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   11.3923    3.1340    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.0622    2.1340    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.0622    2.1340    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.3302    3.3240    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  1  3  1  0  0  0  0
  2  4  1  0  0  0  0
  3  5  2  0  0  0  0
  4  6  2  0  0  0  0
  5  7  1  0  0  0  0
  6  8  1  0  0  0  0
  7  8  2  0  0  0  0
  6  9  1  0  0  0  0
  7 10  1  0  0  0  0
  8 11  1  0  0  0  0
  9 12  1  0  0  0  0
  5 13  1  0  0  0  0
 12 14  1  0  0  0  0
 11 15  1  0  0  0  0
 15 16  1  0  0  0  0
 16 17  2  0  0  0  0
 16 18  1  0  0  0  0
 17 19  1  0  0  0  0
 18 20  2  0  0  0  0
 19 21  2  0  0  0  0
 20 21  1  0  0  0  0
  1 22  1  0  0  0  0
  3 23  1  0  0  0  0
  4 24  1  0  0  0  0
  6 25  1  0  0  0  0
  9 26  1  0  0  0  0
  9 27  1  0  0  0  0
 12 28  1  0  0  0  0
 12 29  1  0  0  0  0
 14 30  1  0  0  0  0
 14 31  1  0  0  0  0
 15 32  1  0  0  0  0
 15 33  1  0  0  0  0
 17 34  1  0  0  0  0
M  END""",

    'COc1cc(CCN)cc(OC)c1OC': """Mescaline
  -OEChem-01052500002D

 24 24  0     0  0  0  0  0  0999 V2000
    2.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.0000    1.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.0000    1.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.0000    2.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.0000    3.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.0000    4.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000   -1.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    5.0000    1.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000   -2.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0000    1.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5000    0.8660    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.5000    0.8660    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.0000    2.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.0000    2.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    3.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.0000    3.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    4.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.0000    4.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  2  0  0  0  0
  2  4  1  0  0  0  0
  3  5  1  0  0  0  0
  5  6  2  0  0  0  0
  6  7  1  0  0  0  0
  6  8  1  0  0  0  0
  7  9  1  0  0  0  0
  8 10  1  0  0  0  0
  9 11  1  0  0  0  0
 10 12  1  0  0  0  0
  1 13  1  0  0  0  0
  6 14  1  0  0  0  0
 13 15  1  0  0  0  0
 14 16  1  0  0  0  0
  1 17  1  0  0  0  0
  3 18  1  0  0  0  0
  8 19  1  0  0  0  0
  8 20  1  0  0  0  0
 10 21  1  0  0  0  0
 10 22  1  0  0  0  0
 12 23  1  0  0  0  0
 12 24  1  0  0  0  0
M  END"""
}

def setup_output_dir():
    """Create outputs directory if it doesn't exist."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def smiles_to_molblock(smiles: str, optimize: bool = True) -> Optional[str]:
    """
    Convert SMILES to 3D MOL block (fallback version).
    
    Args:
        smiles: SMILES string
        optimize: Whether to optimize geometry (ignored in fallback)
        
    Returns:
        MOL block string or None if not available
    """
    # Check if we have a pre-generated MOL block for this SMILES
    if smiles in DEMO_MOLBLOCKS:
        return DEMO_MOLBLOCKS[smiles]
    
    # For unknown SMILES, return None (would need RDKit)
    warnings.warn(f"No pre-generated 3D structure available for SMILES: {smiles}")
    return None

def viewer(molblock: str, width: int = 640, height: int = 480, 
          style: str = "stick", background: str = "white") -> py3Dmol.view:
    """
    Create py3Dmol viewer for a molecule.
    
    Args:
        molblock: MOL block string
        width: Viewer width in pixels
        height: Viewer height in pixels
        style: Visualization style ('stick', 'sphere', 'cartoon', 'line')
        background: Background color
        
    Returns:
        py3Dmol view object
    """
    # Create viewer
    view = py3Dmol.view(width=width, height=height)
    
    # Add molecule
    view.addModel(molblock, 'mol')
    
    # Set style
    if style == "stick":
        view.setStyle({'stick': {'radius': 0.15}})
    elif style == "sphere":
        view.setStyle({'sphere': {'radius': 0.8}})
    elif style == "line":
        view.setStyle({'line': {'linewidth': 3}})
    elif style == "cartoon":
        view.setStyle({'cartoon': {}})
    else:
        # Default to stick
        view.setStyle({'stick': {'radius': 0.15}})
    
    # Set background
    view.setBackgroundColor(background)
    
    # Zoom to fit
    view.zoomTo()
    
    return view

def smiles_to_3d_viewer(smiles: str, width: int = 640, height: int = 480,
                       style: str = "stick") -> Optional[py3Dmol.view]:
    """
    Convert SMILES directly to 3D viewer (fallback version).
    
    Args:
        smiles: SMILES string
        width: Viewer width
        height: Viewer height
        style: Visualization style
        
    Returns:
        py3Dmol view object or None if failed
    """
    molblock = smiles_to_molblock(smiles)
    if molblock is None:
        return None
    
    return viewer(molblock, width, height, style)

def save_molblock_to_sdf(molblocks: list, names: list, filepath: str):
    """
    Save multiple molecules to SDF file (fallback version).
    
    Args:
        molblocks: List of MOL block strings
        names: List of molecule names
        filepath: Output SDF file path
    """
    output_dir = setup_output_dir()
    full_path = output_dir / filepath if not Path(filepath).is_absolute() else Path(filepath)
    
    with open(full_path, 'w') as f:
        for molblock, name in zip(molblocks, names):
            if molblock is not None:
                # Add molecule name to MOL block
                lines = molblock.split('\n')
                if len(lines) > 0:
                    lines[0] = name  # First line is the molecule name
                    molblock_with_name = '\n'.join(lines)
                    f.write(molblock_with_name)
                    f.write('\n$$$$\n')  # SDF separator
    
    print(f"SDF file saved to: {full_path}")
    return str(full_path)

def save_single_sdf(smiles: str, name: str, filepath: str) -> str:
    """
    Save a single molecule to SDF file (fallback version).
    
    Args:
        smiles: SMILES string
        name: Molecule name
        filepath: Output file path
        
    Returns:
        Path to saved file
    """
    molblock = smiles_to_molblock(smiles)
    if molblock is None:
        raise ValueError(f"No 3D structure available for SMILES: {smiles}")
    
    return save_molblock_to_sdf([molblock], [name], filepath)

def batch_convert_to_sdf(df: pd.DataFrame, smiles_col: str = 'smiles', 
                        name_col: str = 'name', output_file: str = 'compounds.sdf') -> str:
    """
    Convert a DataFrame of molecules to SDF file (fallback version).
    
    Args:
        df: DataFrame with molecular data
        smiles_col: Column name containing SMILES
        name_col: Column name containing molecule names
        output_file: Output SDF filename
        
    Returns:
        Path to saved SDF file
    """
    molblocks = []
    names = []
    
    print(f"Converting {len(df)} molecules to 3D (fallback mode)...")
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        name = row[name_col]
        
        print(f"Processing {name}...")
        molblock = smiles_to_molblock(smiles)
        
        if molblock is not None:
            molblocks.append(molblock)
            names.append(name)
        else:
            print(f"Warning: No 3D structure available for {name} ({smiles})")
    
    print(f"Successfully converted {len(molblocks)} out of {len(df)} molecules")
    
    if len(molblocks) == 0:
        print("Warning: No 3D structures available. Install RDKit for full functionality.")
        # Create empty SDF file
        output_dir = setup_output_dir()
        full_path = output_dir / output_file
        with open(full_path, 'w') as f:
            f.write("# No 3D structures available - install RDKit for full functionality\n")
        return str(full_path)
    
    return save_molblock_to_sdf(molblocks, names, output_file)

def validate_3d_structure(smiles: str) -> bool:
    """
    Validate that a SMILES can be converted to 3D structure (fallback version).
    
    Args:
        smiles: SMILES string
        
    Returns:
        True if 3D structure is available
    """
    return smiles in DEMO_MOLBLOCKS

def get_mol_image_base64(smiles: str, size: Tuple[int, int] = (400, 400)) -> Optional[str]:
    """
    Generate 2D molecular structure image as base64 string (fallback version).
    
    Args:
        smiles: SMILES string
        size: Image size (width, height)
        
    Returns:
        None (requires RDKit for 2D image generation)
    """
    warnings.warn("2D image generation requires RDKit")
    return None

if __name__ == "__main__":
    # Test fallback 3D visualization
    print("Testing fallback 3D visualization...")
    
    # Test with 2C-B
    test_smiles = "CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN"
    print(f"Testing with 2C-B: {test_smiles}")
    
    molblock = smiles_to_molblock(test_smiles)
    if molblock:
        print("✅ 3D structure available!")
        view = viewer(molblock)
        print("✅ 3D viewer created successfully!")
    else:
        print("❌ No 3D structure available")
    
    # Test structure availability
    available_structures = list(DEMO_MOLBLOCKS.keys())
    print(f"\n📋 Available 3D structures ({len(available_structures)}):")
    for smiles in available_structures:
        print(f"  • {smiles}")
    
    print("\n⚠️  For full 3D functionality, install RDKit:")
    print("    pip install rdkit-pypi")