"""
5-HT2A Receptor Binding Prediction Module
=========================================

Specialized for CNS therapeutics and psychedelic drug design.
Focuses on 5-HT2A receptor binding affinity prediction and selectivity analysis.

Author: AI Assistant for CNS Therapeutics Research
Target: 5-HT2A receptor-targeted psychedelic therapeutics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
import warnings
warnings.filterwarnings('ignore')

class HT2AReceptorPredictor:
    """Advanced 5-HT2A receptor binding prediction for psychedelic compounds."""
    
    def __init__(self):
        # Known 5-HT2A ligand data for model training/validation
        self.reference_compounds = {
            '2C-B': {'smiles': 'CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN', 'pki': 8.7, 'selectivity': 'high'},
            '2C-I': {'smiles': 'CCc1cc(I)c(OCc2ccccc2)c(I)c1CCN', 'pki': 8.9, 'selectivity': 'high'},
            'DOB': {'smiles': 'CC(N)Cc1cc(Br)c(OCc2ccccc2)c(Br)c1', 'pki': 8.2, 'selectivity': 'moderate'},
            'Mescaline': {'smiles': 'COc1cc(CCN)cc(OC)c1OC', 'pki': 6.2, 'selectivity': 'low'},
            'LSD': {'smiles': 'CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4cccc(c34)C2=C1)C', 'pki': 9.1, 'selectivity': 'high'}
        }
        
        # 5-HT2A receptor pharmacophore features
        self.pharmacophore_features = {
            'aromatic_center': {'weight': 3.0, 'description': 'Aromatic ring system'},
            'basic_nitrogen': {'weight': 2.5, 'description': 'Protonatable nitrogen'},
            'optimal_distance': {'weight': 2.0, 'description': '5-7 Å from aromatic to nitrogen'},
            'halogen_binding': {'weight': 1.5, 'description': 'Halogen bonding (Br, I optimal)'},
            'hydrophobic_region': {'weight': 1.0, 'description': 'Lipophilic substituents'}
        }
    
    def predict_binding_affinity(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Predict 5-HT2A binding affinity using structure-based features.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with binding predictions
        """
        if mol is None:
            return {'pki_predicted': 0.0, 'confidence': 0.0}
        
        # Calculate pharmacophore score
        pharmacophore_score = self._calculate_pharmacophore_match(mol)
        
        # Calculate physicochemical features
        physchem_score = self._calculate_physchem_score(mol)
        
        # Calculate structural similarity to known actives
        similarity_score = self._calculate_similarity_score(mol)
        
        # Combine scores using weighted model
        weights = {'pharmacophore': 0.4, 'physchem': 0.3, 'similarity': 0.3}
        
        combined_score = (
            pharmacophore_score * weights['pharmacophore'] +
            physchem_score * weights['physchem'] +
            similarity_score * weights['similarity']
        )
        
        # Convert to pKi scale (4-10)
        pki_predicted = 4.0 + (combined_score * 6.0)
        
        # Calculate confidence based on score consistency
        confidence = min(1.0, (pharmacophore_score + physchem_score + similarity_score) / 3)
        
        return {
            'pki_predicted': round(pki_predicted, 2),
            'pharmacophore_score': round(pharmacophore_score, 3),
            'physchem_score': round(physchem_score, 3),
            'similarity_score': round(similarity_score, 3),
            'confidence': round(confidence, 3),
            'activity_class': self._classify_activity(pki_predicted),
            'selectivity_prediction': self._predict_selectivity(mol)
        }
    
    def _calculate_pharmacophore_match(self, mol: Chem.Mol) -> float:
        """Calculate how well molecule matches 5-HT2A pharmacophore."""
        score = 0.0
        max_score = sum([feature['weight'] for feature in self.pharmacophore_features.values()])
        
        # Aromatic ring system
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        if aromatic_rings >= 1:
            score += self.pharmacophore_features['aromatic_center']['weight']
        
        # Basic nitrogen
        basic_nitrogens = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0:
                basic_nitrogens += 1
        if basic_nitrogens >= 1:
            score += self.pharmacophore_features['basic_nitrogen']['weight']
        
        # Optimal distance (simplified - would need 3D structure for accuracy)
        if aromatic_rings >= 1 and basic_nitrogens >= 1:
            score += self.pharmacophore_features['optimal_distance']['weight'] * 0.8
        
        # Halogen bonding
        halogens = ['Br', 'I', 'Cl', 'F']
        halogen_count = sum([1 for atom in mol.GetAtoms() if atom.GetSymbol() in halogens])
        if halogen_count > 0:
            bonus = min(1.0, halogen_count / 2.0)  # Optimal with 2 halogens
            score += self.pharmacophore_features['halogen_binding']['weight'] * bonus
        
        # Hydrophobic regions
        logp = Crippen.MolLogP(mol)
        if 2 <= logp <= 5:
            score += self.pharmacophore_features['hydrophobic_region']['weight']
        
        return min(1.0, score / max_score)
    
    def _calculate_physchem_score(self, mol: Chem.Mol) -> float:
        """Calculate physicochemical property score for 5-HT2A binding."""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Optimal ranges for 5-HT2A ligands
        mw_score = 1.0 if 200 <= mw <= 400 else max(0, 1 - abs(mw - 300) / 200)
        logp_score = 1.0 if 2 <= logp <= 5 else max(0, 1 - abs(logp - 3.5) / 3)
        tpsa_score = 1.0 if 20 <= tpsa <= 60 else max(0, 1 - abs(tpsa - 40) / 40)
        
        return (mw_score + logp_score + tpsa_score) / 3
    
    def _calculate_similarity_score(self, mol: Chem.Mol) -> float:
        """Calculate structural similarity to known 5-HT2A actives."""
        from rdkit.Chem import DataStructs
        from rdkit.Chem.Fingerprints import FingerprintMols
        
        query_fp = FingerprintMols.FingerprintMol(mol)
        
        max_similarity = 0.0
        for compound_data in self.reference_compounds.values():
            ref_mol = Chem.MolFromSmiles(compound_data['smiles'])
            if ref_mol:
                ref_fp = FingerprintMols.FingerprintMol(ref_mol)
                similarity = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _classify_activity(self, pki: float) -> str:
        """Classify predicted activity level."""
        if pki >= 8.0:
            return "High Activity"
        elif pki >= 6.5:
            return "Moderate Activity"
        elif pki >= 5.0:
            return "Low Activity"
        else:
            return "Inactive"
    
    def _predict_selectivity(self, mol: Chem.Mol) -> Dict[str, str]:
        """Predict selectivity vs other serotonin receptors."""
        # Simplified selectivity prediction
        tpsa = Descriptors.TPSA(mol)
        logp = Crippen.MolLogP(mol)
        
        # 5-HT2A vs 5-HT2C selectivity
        if tpsa < 50 and 3 <= logp <= 5:
            ht2a_vs_ht2c = "Selective for 5-HT2A"
        else:
            ht2a_vs_ht2c = "May bind 5-HT2C"
        
        # 5-HT2A vs 5-HT1A selectivity
        if logp > 3:
            ht2a_vs_ht1a = "Selective for 5-HT2A"
        else:
            ht2a_vs_ht1a = "May bind 5-HT1A"
        
        return {
            '5-HT2A_vs_5-HT2C': ht2a_vs_ht2c,
            '5-HT2A_vs_5-HT1A': ht2a_vs_ht1a,
            'overall_selectivity': "High" if "Selective" in ht2a_vs_ht2c and "Selective" in ht2a_vs_ht1a else "Moderate"
        }

def analyze_5ht2a_binding_batch(compounds_df: pd.DataFrame) -> pd.DataFrame:
    """Batch analysis of 5-HT2A binding for multiple compounds."""
    predictor = HT2AReceptorPredictor()
    results = []
    
    for idx, row in compounds_df.iterrows():
        smiles = row['smiles']
        name = row.get('name', f'Compound_{idx}')
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                binding_analysis = predictor.predict_binding_affinity(mol)
                
                result = {
                    'name': name,
                    'smiles': smiles,
                    **binding_analysis
                }
                results.append(result)
        except Exception as e:
            print(f"Error analyzing {name}: {str(e)}")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Test with 2C-B
    predictor = HT2AReceptorPredictor()
    test_mol = Chem.MolFromSmiles("CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN")
    
    print("🧬 Testing 5-HT2A Receptor Binding Predictor...")
    results = predictor.predict_binding_affinity(test_mol)
    
    print(f"\n📊 2C-B Binding Prediction:")
    print(f"Predicted pKi: {results['pki_predicted']}")
    print(f"Activity Class: {results['activity_class']}")
    print(f"Confidence: {results['confidence']:.3f}")
    print(f"Overall Selectivity: {results['selectivity_prediction']['overall_selectivity']}")
    print("\n✅ 5-HT2A Binding Predictor Ready!")