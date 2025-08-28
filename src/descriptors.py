"""
Molecular descriptor calculations for psychedelic compounds.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import RDKit, fall back to mock functions if not available
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not available. Using fallback molecular property calculations.")
    # Import fallback functions
    from descriptors_fallback import (
        calculate_descriptors_fallback, get_known_properties, 
        enhance_with_known_properties, KNOWN_PROPERTIES
    )

def smiles_to_mol(smiles: str) -> Any:
    """
    Convert SMILES string to RDKit molecule object.
    
    Args:
        smiles: SMILES string
        
    Returns:
        RDKit molecule object or None if invalid
    """
    if not RDKIT_AVAILABLE:
        # Fallback: simple SMILES validation
        if isinstance(smiles, str) and len(smiles) > 3 and 'C' in smiles:
            return smiles  # Return SMILES as "mol" object
        return None
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None

def calculate_descriptors(mol: Any) -> Dict[str, Any]:
    """
    Calculate molecular descriptors for a single molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary of molecular descriptors
    """
    if not RDKIT_AVAILABLE:
        # Use fallback calculation
        if isinstance(mol, str):  # mol is actually SMILES in fallback mode
            return calculate_descriptors_fallback(mol)
        return {
            'mw': np.nan, 'logp': np.nan, 'tpsa': np.nan,
            'hbd': np.nan, 'hba': np.nan, 'rotb': np.nan, 'rings': np.nan
        }
    
    if mol is None:
        return {
            'mw': np.nan, 'logp': np.nan, 'tpsa': np.nan,
            'hbd': np.nan, 'hba': np.nan, 'rotb': np.nan, 'rings': np.nan
        }
    
    try:
        descriptors = {
            'mw': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rotb': Descriptors.NumRotatableBonds(mol),
            'rings': Descriptors.RingCount(mol)
        }
        return descriptors
    except:
        return {
            'mw': np.nan, 'logp': np.nan, 'tpsa': np.nan,
            'hbd': np.nan, 'hba': np.nan, 'rotb': np.nan, 'rings': np.nan
        }

def calculate_drug_likeness(descriptors: Dict[str, Any]) -> float:
    """
    Calculate simple drug-likeness score based on Lipinski's Rule of Five.
    
    Args:
        descriptors: Dictionary of molecular descriptors
        
    Returns:
        Drug-likeness score (0-1, higher is better)
    """
    violations = 0
    
    # Check Lipinski's Rule of Five
    if descriptors['mw'] > 500:
        violations += 1
    if descriptors['logp'] > 5:
        violations += 1
    if descriptors['hbd'] > 5:
        violations += 1
    if descriptors['hba'] > 10:
        violations += 1
    
    # Convert violations to score (0-4 violations -> 1.0-0.0 score)
    drug_likeness = max(0.0, 1.0 - (violations / 4.0))
    
    return drug_likeness

def bbb_label(tpsa: float) -> str:
    """
    Predict blood-brain barrier penetration based on TPSA.
    
    Args:
        tpsa: Topological polar surface area
        
    Returns:
        "Good BBB" or "Poor BBB"
    """
    if pd.isna(tpsa):
        return "Unknown"
    
    # TPSA < 60 Ų is generally favorable for BBB penetration
    return "Good BBB" if tpsa < 60 else "Poor BBB"

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute molecular descriptors for all compounds in the DataFrame.
    
    Args:
        df: DataFrame with 'smiles' column
        
    Returns:
        DataFrame with added descriptor columns
    """
    df_copy = df.copy()
    
    # Initialize descriptor columns
    descriptor_cols = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotb', 'rings']
    for col in descriptor_cols:
        df_copy[col] = np.nan
    
    df_copy['drug_likeness'] = np.nan
    df_copy['bbb_label'] = "Unknown"
    
    # Calculate descriptors for each molecule
    for idx, row in df_copy.iterrows():
        smiles = row['smiles']
        mol = smiles_to_mol(smiles)
        
        if mol is not None:
            # Calculate descriptors
            descriptors = calculate_descriptors(mol)
            
            # Update DataFrame
            for col in descriptor_cols:
                df_copy.loc[idx, col] = descriptors[col]
            
            # Calculate drug-likeness
            df_copy.loc[idx, 'drug_likeness'] = calculate_drug_likeness(descriptors)
            
            # Calculate BBB label
            df_copy.loc[idx, 'bbb_label'] = bbb_label(descriptors['tpsa'])
    
    # Enhance with known properties if RDKit is not available
    if not RDKIT_AVAILABLE:
        df_copy = enhance_with_known_properties(df_copy)
    
    return df_copy

def get_descriptor_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Get summary statistics for molecular descriptors.
    
    Args:
        df: DataFrame with descriptor columns
        
    Returns:
        Dictionary of statistics for each descriptor
    """
    descriptor_cols = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotb', 'rings', 'drug_likeness']
    stats = {}
    
    for col in descriptor_cols:
        if col in df.columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
    
    return stats

if __name__ == "__main__":
    # Test descriptor calculations
    from data import load_demo
    
    print(f"RDKit available: {RDKIT_AVAILABLE}")
    
    df = load_demo()
    print(f"Loaded {len(df)} compounds")
    
    # Compute descriptors
    df_with_descriptors = compute(df)
    print("\nDescriptors computed successfully!")
    
    # Show sample results
    print("\nSample results:")
    print(df_with_descriptors[['name', 'mw', 'logp', 'tpsa', 'drug_likeness', 'bbb_label']].head())
    
    # Show statistics
    stats = get_descriptor_stats(df_with_descriptors)
    print(f"\nMolecular weight range: {stats['mw']['min']:.1f} - {stats['mw']['max']:.1f} Da")
    print(f"Average drug-likeness: {stats['drug_likeness']['mean']:.2f}")
    
    # BBB distribution
    bbb_counts = df_with_descriptors['bbb_label'].value_counts()
    print(f"\nBBB penetration prediction:")
    for label, count in bbb_counts.items():
        print(f"  {label}: {count} compounds")