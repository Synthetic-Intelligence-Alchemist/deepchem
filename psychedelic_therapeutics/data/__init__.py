"""
Data Collection and Curation Module
===================================

Tools for collecting and curating 2C-series psychedelic compounds
from various sources including ChEMBL, PubChem, and literature.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import deepchem as dc
from chembl_webresource_client.new_client import new_client

class PsychedelicDataCollector:
    """Collect and curate psychedelic compound data."""
    
    def __init__(self):
        self.molecule_client = new_client.molecule
        self.activity_client = new_client.activity
        self.target_client = new_client.target
        
    def get_2c_series_smiles(self) -> Dict[str, str]:
        """Get SMILES for known 2C-series compounds."""
        compounds = {
            # Core 2C compounds
            '2C-B': 'CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN',  # Corrected structure
            '2C-I': 'CCc1cc(I)c(OCc2ccccc2)c(I)c1CCN',
            '2C-E': 'CCCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN',
            '2C-P': 'CC(C)c1cc(Br)c(OCc2ccccc2)c(Br)c1CCN',
            '2C-T-2': 'CCc1cc(SCc2ccccc2)c(OCc3ccccc3)c(SCc4ccccc4)c1CCN',
            '2C-T-4': 'CCc1cc(SC(C)C)c(OCc2ccccc2)c(SC(C)C)c1CCN',
            
            # DOx series (amphetamine versions)
            'DOB': 'CC(N)Cc1cc(Br)c(OCc2ccccc2)c(Br)c1',
            'DOI': 'CC(N)Cc1cc(I)c(OCc2ccccc2)c(I)c1',
            'DOM': 'COc1cc(CC(C)N)cc(OC)c1OCc1ccccc1',
            
            # NBOMe series
            '25B-NBOMe': 'COc1cc(CCNCc2ccccc2OC)c(Br)cc1OCc1ccccc1',
            '25I-NBOMe': 'COc1cc(CCNCc2ccccc2OC)c(I)cc1OCc1ccccc1',
            
            # Mescaline and analogs
            'Mescaline': 'COc1cc(CCN)cc(OC)c1OC',
            'Escaline': 'CCOc1cc(CCN)cc(OCC)c1OCC',
            'Proscaline': 'CCCOc1cc(CCN)cc(OCCC)c1OCCC',
        }
        
        # Validate SMILES
        validated_compounds = {}
        for name, smiles in compounds.items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                validated_compounds[name] = Chem.MolToSmiles(mol)
            else:
                print(f"Warning: Invalid SMILES for {name}: {smiles}")
                
        return validated_compounds
    
    def search_chembl_5ht2a(self, limit: int = 1000) -> pd.DataFrame:
        """Search ChEMBL for 5-HT2A receptor binding data."""
        try:
            # Search for 5-HT2A receptor
            targets = self.target_client.filter(
                target_synonym__icontains='5-HT2A'
            ).only(['target_chembl_id', 'pref_name', 'target_type'])
            
            activities_data = []
            
            for target in targets:
                # Get activities for this target
                activities = self.activity_client.filter(
                    target_chembl_id=target['target_chembl_id'],
                    standard_type__in=['Ki', 'Kd', 'IC50', 'EC50'],
                    standard_value__isnull=False
                ).only([
                    'molecule_chembl_id', 'standard_type', 'standard_value',
                    'standard_units', 'pchembl_value', 'canonical_smiles'
                ])[:limit]
                
                for activity in activities:
                    if activity['canonical_smiles']:
                        activities_data.append({
                            'molecule_chembl_id': activity['molecule_chembl_id'],
                            'smiles': activity['canonical_smiles'],
                            'target_chembl_id': target['target_chembl_id'],
                            'target_name': target['pref_name'],
                            'assay_type': activity['standard_type'],
                            'value': activity['standard_value'],
                            'units': activity['standard_units'],
                            'pchembl_value': activity['pchembl_value']
                        })
            
            return pd.DataFrame(activities_data)
            
        except Exception as e:
            print(f"Error accessing ChEMBL: {e}")
            return pd.DataFrame()
    
    def calculate_molecular_properties(self, smiles: str) -> Dict[str, float]:
        """Calculate molecular properties for a given SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
            
        return {
            'mw': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'tpsa': Descriptors.TPSA(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol)
        }
    
    def load_toxcast_5ht2a(self) -> Tuple[List[str], Tuple, List]:
        """Load 5-HT2A data from ToxCast via DeepChem."""
        try:
            tasks, datasets, transformers = dc.molnet.load_toxcast(
                featurizer='ECFP',
                tasks=['NVS_GPCR_h5HT2A'],
                reload=True
            )
            return tasks, datasets, transformers
        except Exception as e:
            print(f"Error loading ToxCast data: {e}")
            return [], (), []
    
    def create_psychedelic_dataset(self) -> pd.DataFrame:
        """Create comprehensive psychedelic dataset."""
        # Get 2C-series compounds
        compounds_2c = self.get_2c_series_smiles()
        
        # Search ChEMBL for 5-HT2A data
        chembl_data = self.search_chembl_5ht2a()
        
        # Create dataset
        dataset_rows = []
        
        # Add 2C-series compounds
        for name, smiles in compounds_2c.items():
            props = self.calculate_molecular_properties(smiles)
            row = {
                'compound_name': name,
                'smiles': smiles,
                'compound_class': '2C-series' if name.startswith('2C') else 
                                'DOx-series' if name.startswith('DO') else
                                'NBOMe-series' if 'NBOMe' in name else
                                'Mescaline-analog',
                'is_psychedelic': True,
                **props
            }
            dataset_rows.append(row)
        
        # Add ChEMBL data
        if not chembl_data.empty:
            for _, row in chembl_data.iterrows():
                props = self.calculate_molecular_properties(row['smiles'])
                dataset_row = {
                    'compound_name': row['molecule_chembl_id'],
                    'smiles': row['smiles'],
                    'compound_class': 'ChEMBL_5HT2A',
                    'is_psychedelic': False,
                    'binding_value': row['value'],
                    'binding_type': row['assay_type'],
                    'pchembl_value': row['pchembl_value'],
                    **props
                }
                dataset_rows.append(dataset_row)
        
        return pd.DataFrame(dataset_rows)

def load_psychedelic_data() -> pd.DataFrame:
    """Convenience function to load psychedelic dataset."""
    collector = PsychedelicDataCollector()
    return collector.create_psychedelic_dataset()

if __name__ == "__main__":
    # Test the data collection
    collector = PsychedelicDataCollector()
    dataset = collector.create_psychedelic_dataset()
    print(f"Created dataset with {len(dataset)} compounds")
    print(dataset.head())