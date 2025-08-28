"""
Data loading utilities for psychedelic compounds.
"""

import pandas as pd
import os
from pathlib import Path

def load_demo() -> pd.DataFrame:
    """
    Load the demo dataset of psychedelic compounds.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['class', 'name', 'smiles']
    """
    # Get the project root directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    data_file = project_root / "data" / "compounds_demo.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Demo data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Validate required columns
    required_cols = ['class', 'name', 'smiles']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

def load_custom_csv(file_path: str) -> pd.DataFrame:
    """
    Load a custom CSV file with psychedelic compounds.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame with molecular data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df

def validate_smiles_column(df: pd.DataFrame, smiles_col: str = 'smiles') -> bool:
    """
    Validate that the DataFrame has a SMILES column with valid entries.
    
    Args:
        df: Input DataFrame
        smiles_col: Name of the SMILES column
        
    Returns:
        bool: True if valid, raises exception if not
    """
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in DataFrame")
    
    if df[smiles_col].isna().any():
        raise ValueError("Found NaN values in SMILES column")
    
    if df[smiles_col].str.strip().eq('').any():
        raise ValueError("Found empty SMILES strings")
    
    return True

if __name__ == "__main__":
    # Test the data loading
    df = load_demo()
    print(f"Loaded {len(df)} compounds")
    print(df.head())
    validate_smiles_column(df)
    print("Data validation passed!")