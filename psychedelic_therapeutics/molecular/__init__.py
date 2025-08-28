"""
Molecular Featurization Module
==============================

Advanced molecular featurization tools optimized for psychedelic compounds
and 5-HT2A receptor binding prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import deepchem as dc
from deepchem.feat import CircularFingerprint, MolGraphConvFeaturizer
from deepchem.feat import RDKitDescriptors, MACCSKeysFingerprint

class PsychedelicFeaturizer:
    """Advanced featurization for psychedelic compounds."""
    
    def __init__(self, featurizer_type: str = 'graph'):
        """
        Initialize featurizer.
        
        Args:
            featurizer_type: Type of featurizer ('ecfp', 'graph', 'rdkit', 'maccs', 'combined')
        """
        self.featurizer_type = featurizer_type
        self._setup_featurizers()
    
    def _setup_featurizers(self):
        """Setup DeepChem featurizers."""
        self.featurizers = {
            'ecfp': CircularFingerprint(size=2048, radius=2),
            'graph': MolGraphConvFeaturizer(use_edges=True, use_chirality=True),
            'rdkit': RDKitDescriptors(),
            'maccs': MACCSKeysFingerprint(),
        }
    
    def calculate_psychedelic_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate descriptors relevant to psychedelic activity."""
        if mol is None:
            return {}
        
        descriptors = {
            # Basic properties
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            
            # Hydrogen bonding
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            
            # Structural features
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol),
            
            # Psychedelic-specific features
            'phenethylamine_core': self._has_phenethylamine_core(mol),
            'methoxy_groups': self._count_methoxy_groups(mol),
            'halogen_substitution': self._count_halogens(mol),
            'benzyl_ether': self._has_benzyl_ether(mol),
            
            # BBB penetration indicators
            'mw_bbb_favorable': 1.0 if Descriptors.MolWt(mol) < 450 else 0.0,
            'logp_bbb_favorable': 1.0 if 1.0 < Descriptors.MolLogP(mol) < 3.0 else 0.0,
            'tpsa_bbb_favorable': 1.0 if Descriptors.TPSA(mol) < 90 else 0.0,
            
            # Drug-likeness
            'lipinski_violations': self._count_lipinski_violations(mol),
            'qed': self._calculate_qed(mol),
        }
        
        return descriptors
    
    def _has_phenethylamine_core(self, mol: Chem.Mol) -> float:
        """Check for phenethylamine core structure."""
        # SMARTS pattern for phenethylamine: benzene ring connected to ethylamine
        phenethylamine_pattern = Chem.MolFromSmarts('c1ccccc1CCN')
        return 1.0 if mol.HasSubstructMatch(phenethylamine_pattern) else 0.0
    
    def _count_methoxy_groups(self, mol: Chem.Mol) -> int:
        """Count methoxy groups (-OCH3)."""
        methoxy_pattern = Chem.MolFromSmarts('COc')
        return len(mol.GetSubstructMatches(methoxy_pattern))
    
    def _count_halogens(self, mol: Chem.Mol) -> int:
        """Count halogen atoms (F, Cl, Br, I)."""
        halogen_count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['F', 'Cl', 'Br', 'I']:
                halogen_count += 1
        return halogen_count
    
    def _has_benzyl_ether(self, mol: Chem.Mol) -> float:
        """Check for benzyl ether group (common in 2C compounds)."""
        benzyl_ether_pattern = Chem.MolFromSmarts('COCc1ccccc1')
        return 1.0 if mol.HasSubstructMatch(benzyl_ether_pattern) else 0.0
    
    def _count_lipinski_violations(self, mol: Chem.Mol) -> int:
        """Count Lipinski rule violations."""
        violations = 0
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        if Descriptors.MolLogP(mol) > 5:
            violations += 1
        if Descriptors.NumHDonors(mol) > 5:
            violations += 1
        if Descriptors.NumHAcceptors(mol) > 10:
            violations += 1
        return violations
    
    def _calculate_qed(self, mol: Chem.Mol) -> float:
        """Calculate QED (Quantitative Estimate of Drug-likeness)."""
        try:
            from rdkit.Chem import QED
            return QED.qed(mol)
        except ImportError:
            return 0.0
    
    def featurize_molecule(self, smiles: str) -> Optional[np.ndarray]:
        """Featurize a single molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        if self.featurizer_type == 'combined':
            return self._combined_featurization(smiles)
        else:
            featurizer = self.featurizers.get(self.featurizer_type)
            if featurizer is None:
                raise ValueError(f"Unknown featurizer type: {self.featurizer_type}")
            
            features = featurizer.featurize([smiles])
            return features[0] if len(features) > 0 else None
    
    def _combined_featurization(self, smiles: str) -> Optional[np.ndarray]:
        """Combine multiple featurization methods."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Get ECFP fingerprints
        ecfp_features = self.featurizers['ecfp'].featurize([smiles])[0]
        
        # Get RDKit descriptors
        rdkit_features = self.featurizers['rdkit'].featurize([smiles])[0]
        
        # Get psychedelic-specific descriptors
        psychedelic_descriptors = self.calculate_psychedelic_descriptors(mol)
        psychedelic_features = np.array(list(psychedelic_descriptors.values()))
        
        # Combine all features
        combined_features = np.concatenate([
            ecfp_features.flatten(),
            rdkit_features.flatten(),
            psychedelic_features.flatten()
        ])
        
        return combined_features
    
    def featurize_dataset(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Featurize a dataset of molecules."""
        features = []
        valid_smiles = []
        
        for smiles in smiles_list:
            feature = self.featurize_molecule(smiles)
            if feature is not None:
                features.append(feature)
                valid_smiles.append(smiles)
        
        if not features:
            return np.array([]), []
        
        # Handle different feature shapes
        if self.featurizer_type == 'graph':
            # Graph features need special handling
            return features, valid_smiles
        else:
            return np.array(features), valid_smiles
    
    def create_deepchem_dataset(self, df: pd.DataFrame, 
                               target_column: str = 'pchembl_value') -> dc.data.Dataset:
        """Create DeepChem dataset from pandas DataFrame."""
        # Extract SMILES and targets
        smiles = df['smiles'].tolist()
        
        if target_column in df.columns:
            targets = df[target_column].values
            # Handle missing values
            targets = np.where(pd.isna(targets), 0, targets)
        else:
            targets = np.zeros(len(smiles))
        
        # Featurize molecules
        if self.featurizer_type == 'graph':
            featurizer = self.featurizers['graph']
            features = featurizer.featurize(smiles)
            dataset = dc.data.DiskDataset.from_numpy(features, targets)
        else:
            features, valid_smiles = self.featurize_dataset(smiles)
            if len(features) == 0:
                raise ValueError("No valid molecules could be featurized")
            
            # Ensure targets match valid molecules
            valid_indices = [i for i, s in enumerate(smiles) if s in valid_smiles]
            valid_targets = targets[valid_indices]
            
            dataset = dc.data.DiskDataset.from_numpy(features, valid_targets)
        
        return dataset

class TwoC_B_Analyzer:
    """Specialized analyzer for 2C-B and its analogs."""
    
    def __init__(self):
        self.featurizer = PsychedelicFeaturizer('combined')
    
    def analyze_2cb_structure(self, smiles: str = 'CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN') -> Dict:
        """Analyze 2C-B structure and properties."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        analysis = {
            'smiles': smiles,
            'mol_formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
            'descriptors': self.featurizer.calculate_psychedelic_descriptors(mol),
            'structural_features': self._analyze_2cb_features(mol),
        }
        
        return analysis
    
    def _analyze_2cb_features(self, mol: Chem.Mol) -> Dict:
        """Analyze specific structural features of 2C-B."""
        features = {
            'bromine_atoms': len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'Br']),
            'benzyl_ether_present': self.featurizer._has_benzyl_ether(mol),
            'phenethylamine_core': self.featurizer._has_phenethylamine_core(mol),
            'primary_amine': self._has_primary_amine(mol),
            'symmetrical_substitution': self._check_symmetrical_bromine(mol),
        }
        return features
    
    def _has_primary_amine(self, mol: Chem.Mol) -> float:
        """Check for primary amine group."""
        primary_amine_pattern = Chem.MolFromSmarts('[NH2]')
        return 1.0 if mol.HasSubstructMatch(primary_amine_pattern) else 0.0
    
    def _check_symmetrical_bromine(self, mol: Chem.Mol) -> float:
        """Check if bromine atoms are symmetrically positioned."""
        # This is a simplified check - in reality, would need more sophisticated analysis
        bromine_pattern = Chem.MolFromSmarts('Brc1cc(*)cc(Br)c1')
        return 1.0 if mol.HasSubstructMatch(bromine_pattern) else 0.0
    
    def compare_with_analogs(self, analog_smiles_list: List[str]) -> pd.DataFrame:
        """Compare 2C-B with its analogs."""
        # 2C-B reference
        cb_analysis = self.analyze_2cb_structure()
        
        results = [cb_analysis]
        
        # Analyze analogs
        for smiles in analog_smiles_list:
            analysis = self.analyze_2cb_structure(smiles)
            results.append(analysis)
        
        # Convert to DataFrame for easier analysis
        df = pd.json_normalize(results)
        return df

def create_psychedelic_featurizer(featurizer_type: str = 'graph') -> PsychedelicFeaturizer:
    """Factory function to create psychedelic featurizer."""
    return PsychedelicFeaturizer(featurizer_type)

if __name__ == "__main__":
    # Test the featurizer
    analyzer = TwoC_B_Analyzer()
    analysis = analyzer.analyze_2cb_structure()
    print("2C-B Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")