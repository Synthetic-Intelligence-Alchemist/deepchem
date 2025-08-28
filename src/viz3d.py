"""
3D molecular visualization using RDKit and py3Dmol.
"""

import pandas as pd
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import RDKit, fall back to fallback functions if not available
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not available. Using fallback 3D visualization.")
    # Import fallback functions
    from viz3d_fallback import (
        smiles_to_molblock as smiles_to_molblock_fallback, 
        viewer as viewer_fallback,
        smiles_to_3d_viewer as smiles_to_3d_viewer_fallback, 
        DEMO_MOLBLOCKS
    )

import py3Dmol
from pathlib import Path
import io
import base64

def setup_output_dir():
    """Create outputs directory if it doesn't exist."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def smiles_to_molblock(smiles: str, optimize: bool = True) -> Optional[str]:
    """
    Convert SMILES to 3D MOL block using RDKit.
    
    Args:
        smiles: SMILES string
        optimize: Whether to optimize geometry with UFF
        
    Returns:
        MOL block string or None if failed
    """
    if not RDKIT_AVAILABLE:
        return smiles_to_molblock_fallback(smiles, optimize)
        
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates using ETKDG
        # ETKDG is a distance geometry method that works well for drug-like molecules
        params = AllChem.EmbedParameters()
        params.useRandomCoords = True
        params.boxSizeMult = 2.0
        params.randNegEig = True
        
        # Try to embed the molecule
        embed_result = AllChem.EmbedMolecule(mol, params)
        
        if embed_result == -1:
            # If ETKDG fails, try with different parameters
            params.useRandomCoords = False
            embed_result = AllChem.EmbedMolecule(mol, params)
            
        if embed_result == -1:
            # If still fails, try basic embedding
            embed_result = AllChem.EmbedMolecule(mol)
            
        if embed_result == -1:
            print(f"Warning: Could not generate 3D coordinates for SMILES: {smiles}")
            return None
        
        # Optimize geometry with UFF (Universal Force Field)
        if optimize:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except:
                print(f"Warning: UFF optimization failed for SMILES: {smiles}")
        
        # Convert to MOL block
        molblock = Chem.MolToMolBlock(mol)
        return molblock
        
    except Exception as e:
        print(f"Error converting SMILES to 3D: {smiles}, Error: {str(e)}")
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
    if not RDKIT_AVAILABLE:
        return viewer_fallback(molblock, width, height, style, background)
        
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
    Convert SMILES directly to 3D viewer.
    
    Args:
        smiles: SMILES string
        width: Viewer width
        height: Viewer height
        style: Visualization style
        
    Returns:
        py3Dmol view object or None if failed
    """
    if not RDKIT_AVAILABLE:
        return smiles_to_3d_viewer_fallback(smiles, width, height, style)
        
    molblock = smiles_to_molblock(smiles)
    if molblock is None:
        return None
    
    return viewer(molblock, width, height, style)

def save_molblock_to_sdf(molblocks: list, names: list, filepath: str):
    """
    Save multiple molecules to SDF file.
    
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
    Save a single molecule to SDF file.
    
    Args:
        smiles: SMILES string
        name: Molecule name
        filepath: Output file path
        
    Returns:
        Path to saved file
    """
    molblock = smiles_to_molblock(smiles)
    if molblock is None:
        raise ValueError(f"Could not generate 3D structure for SMILES: {smiles}")
    
    return save_molblock_to_sdf([molblock], [name], filepath)

def get_mol_image_base64(smiles: str, size: Tuple[int, int] = (400, 400)) -> Optional[str]:
    """
    Generate 2D molecular structure image as base64 string.
    
    Args:
        smiles: SMILES string
        size: Image size (width, height)
        
    Returns:
        Base64 encoded PNG image or None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate 2D coordinates
        rdDepictor.Compute2DCoords(mol)
        
        # Draw molecule
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Get image data
        img_data = drawer.GetDrawingText()
        
        # Encode as base64
        img_b64 = base64.b64encode(img_data).decode()
        return img_b64
        
    except Exception as e:
        print(f"Error generating 2D image for SMILES: {smiles}, Error: {str(e)}")
        return None

def batch_convert_to_sdf(df: pd.DataFrame, smiles_col: str = 'smiles', 
                        name_col: str = 'name', output_file: str = 'compounds.sdf') -> str:
    """
    Convert a DataFrame of molecules to SDF file.
    
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
    
    print(f"Converting {len(df)} molecules to 3D...")
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        name = row[name_col]
        
        print(f"Processing {name}...")
        molblock = smiles_to_molblock(smiles)
        
        if molblock is not None:
            molblocks.append(molblock)
            names.append(name)
        else:
            print(f"Warning: Failed to convert {name} ({smiles})")
    
    print(f"Successfully converted {len(molblocks)} out of {len(df)} molecules")
    
    return save_molblock_to_sdf(molblocks, names, output_file)

def validate_3d_structure(smiles: str) -> bool:
    """
    Validate that a SMILES can be converted to 3D structure.
    
    Args:
        smiles: SMILES string
        
    Returns:
        True if 3D structure can be generated
    """
    if not RDKIT_AVAILABLE:
        # Basic validation for fallback mode
        return isinstance(smiles, str) and len(smiles) > 3 and 'C' in smiles
        
    molblock = smiles_to_molblock(smiles)
    return molblock is not None

if __name__ == "__main__":
    # Test 3D visualization
    from data import load_demo
    
    print("Loading demo data...")
    df = load_demo()
    
    # Test single molecule conversion
    test_smiles = df.iloc[0]['smiles']
    test_name = df.iloc[0]['name']
    
    print(f"Testing 3D conversion for {test_name}...")
    molblock = smiles_to_molblock(test_smiles)
    
    if molblock:
        print("✅ 3D conversion successful!")
        
        # Test viewer creation
        view = viewer(molblock)
        print("✅ 3D viewer created successfully!")
        
        # Test SDF export
        sdf_path = save_single_sdf(test_smiles, test_name, "test_molecule.sdf")
        print(f"✅ SDF export successful: {sdf_path}")
        
    else:
        print("❌ 3D conversion failed")
    
    # Test batch conversion
    print(f"\nTesting batch conversion for all {len(df)} molecules...")
    try:
        batch_sdf_path = batch_convert_to_sdf(df)
        print(f"✅ Batch conversion successful: {batch_sdf_path}")
    except Exception as e:
        print(f"❌ Batch conversion failed: {str(e)}")
    
    print("\n3D visualization tests completed!")