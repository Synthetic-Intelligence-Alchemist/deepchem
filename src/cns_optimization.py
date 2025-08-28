"""
CNS Optimization Tools for Psychedelic Therapeutics
===================================================

Advanced scoring and optimization for CNS penetration, pharmacokinetics,
and drug-like properties specifically for 5-HT2A receptor-targeted compounds.

Author: AI Assistant for CNS Therapeutics Research
Focus: Psychedelic drug optimization for therapeutic applications
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
import math

class CNSOptimizer:
    """Advanced CNS optimization for psychedelic therapeutics."""
    
    def __init__(self):
        # CNS-specific parameter ranges optimized for psychedelics
        self.optimal_ranges = {
            'mw': (180, 450),      # Molecular weight for BBB penetration
            'logp': (1.5, 4.5),    # Lipophilicity for membrane permeation
            'tpsa': (20, 65),      # Polar surface area for BBB
            'hbd': (0, 2),         # H-bond donors
            'hba': (1, 6),         # H-bond acceptors
            'rotb': (0, 8),        # Rotatable bonds for flexibility
            'clogp': (2, 4),       # Calculated LogP
            'aromatic_rings': (1, 3)  # Aromatic rings for CNS activity
        }
        
        # Psychedelic-specific CNS factors
        self.psychedelic_factors = {
            'ht2a_selectivity': 2.0,    # Bonus for 5-HT2A selectivity
            'halogen_bonus': 1.5,       # Halogen substitution benefit
            'phenethylamine_bonus': 1.8, # 2C-type scaffold bonus
            'safety_penalty': -2.0      # Penalty for toxic features
        }
    
    def calculate_cns_score(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate comprehensive CNS optimization score.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with CNS optimization metrics
        """
        if mol is None:
            return {}
        
        # Basic descriptors
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotb = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        
        # Calculate individual scores
        scores = {
            'bbb_score': self._calculate_bbb_score(mw, logp, tpsa, hbd),
            'drug_likeness_score': self._calculate_drug_likeness_score(mol),
            'cns_mpo_score': self._calculate_cns_mpo_score(mol),
            'permeability_score': self._calculate_permeability_score(mol),
            'metabolic_stability_score': self._calculate_metabolic_stability(mol),
            'safety_score': self._calculate_safety_score(mol),
            'psychedelic_profile_score': self._calculate_psychedelic_profile(mol)
        }
        
        # Calculate composite CNS score
        weights = {
            'bbb_score': 0.25,
            'drug_likeness_score': 0.15,
            'cns_mpo_score': 0.20,
            'permeability_score': 0.15,
            'metabolic_stability_score': 0.10,
            'safety_score': 0.10,
            'psychedelic_profile_score': 0.05
        }
        
        composite_score = sum(scores[key] * weights[key] for key in weights.keys())
        scores['composite_cns_score'] = min(1.0, max(0.0, composite_score))
        
        # Add optimization recommendations
        scores['optimization_recommendations'] = self._generate_optimization_recommendations(mol, scores)
        scores['cns_classification'] = self._classify_cns_potential(scores['composite_cns_score'])
        
        return scores
    
    def _calculate_bbb_score(self, mw: float, logp: float, tpsa: float, hbd: int) -> float:
        """Enhanced BBB penetration prediction for psychedelics."""
        score = 1.0
        
        # Molecular weight penalty
        if mw > 450:
            score *= (500 - mw) / 50
        elif mw < 150:
            score *= mw / 150
        
        # LogP optimization (psychedelics need moderate lipophilicity)
        if logp < 1:
            score *= logp
        elif logp > 5:
            score *= (6 - logp)
        elif 2 <= logp <= 4:
            score *= 1.2  # Bonus for optimal range
        
        # TPSA optimization (critical for BBB)
        if tpsa > 90:
            score *= (120 - tpsa) / 30
        elif tpsa < 20:
            score *= tpsa / 20
        elif tpsa <= 60:
            score *= 1.3  # Bonus for good BBB range
        
        # H-bond donor penalty
        if hbd > 2:
            score *= (5 - hbd) / 3
        
        return min(1.0, max(0.0, score))
    
    def _calculate_drug_likeness_score(self, mol: Chem.Mol) -> float:
        """Calculate drug-likeness optimized for CNS compounds."""
        violations = 0
        
        # Lipinski violations
        if Descriptors.MolWt(mol) > 500: violations += 1
        if Crippen.MolLogP(mol) > 5: violations += 1
        if Descriptors.NumHDonors(mol) > 5: violations += 1
        if Descriptors.NumHAcceptors(mol) > 10: violations += 1
        
        # CNS-specific rules
        if Descriptors.TPSA(mol) > 90: violations += 1
        if Descriptors.NumRotatableBonds(mol) > 10: violations += 1
        
        # Veber rules for oral bioavailability
        if Descriptors.TPSA(mol) > 140: violations += 1
        if Descriptors.NumRotatableBonds(mol) > 10: violations += 1
        
        # Convert violations to score
        max_violations = 8
        score = max(0.0, (max_violations - violations) / max_violations)
        
        return score
    
    def _calculate_cns_mpo_score(self, mol: Chem.Mol) -> float:
        """Calculate CNS Multiparameter Optimization (MPO) score."""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        tpsa = Descriptors.TPSA(mol)
        logd = logp  # Simplified - would need pH calculation for true LogD
        pka = 9.0    # Assumed basic pKa for psychedelics
        
        # CNS MPO scoring functions
        def mw_score(mw):
            if mw <= 360: return 1.0
            elif mw <= 500: return 1.0 - (mw - 360) / 140
            else: return 0.0
        
        def logp_score(logp):
            if 1 <= logp <= 3: return 1.0
            elif logp < 1: return logp
            elif logp <= 5: return 1.0 - (logp - 3) / 2
            else: return 0.0
        
        def hbd_score(hbd):
            if hbd <= 0.5: return 1.0
            elif hbd <= 3.5: return 1.0 - (hbd - 0.5) / 3
            else: return 0.0
        
        def tpsa_score(tpsa):
            if tpsa <= 40: return 1.0
            elif tpsa <= 90: return 1.0 - (tpsa - 40) / 50
            else: return 0.0
        
        def logd_score(logd):
            if 1 <= logd <= 3: return 1.0
            elif logd < 1: return logd
            elif logd <= 4: return 1.0 - (logd - 3) / 1
            else: return 0.0
        
        def pka_score(pka):
            if 7.5 <= pka <= 10.5: return 1.0
            elif pka < 7.5: return max(0, pka / 7.5)
            else: return max(0, (12 - pka) / 1.5)
        
        # Calculate individual scores
        scores = [
            mw_score(mw),
            logp_score(logp),
            hbd_score(hbd),
            tpsa_score(tpsa),
            logd_score(logd),
            pka_score(pka)
        ]
        
        # CNS MPO is sum of individual scores (0-6 scale)
        cns_mpo = sum(scores)
        
        # Normalize to 0-1 scale
        return cns_mpo / 6.0
    
    def _calculate_permeability_score(self, mol: Chem.Mol) -> float:
        """Calculate membrane permeability score."""
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
        
        # Permeability factors
        lipophilicity_factor = 1.0 if 2 <= logp <= 4 else max(0, 1 - abs(logp - 3) / 3)
        size_factor = 1.0 if mw <= 400 else max(0, (500 - mw) / 100)
        polarity_factor = 1.0 if tpsa <= 60 else max(0, (90 - tpsa) / 30)
        
        return (lipophilicity_factor + size_factor + polarity_factor) / 3
    
    def _calculate_metabolic_stability(self, mol: Chem.Mol) -> float:
        """Estimate metabolic stability based on structural features."""
        score = 1.0
        
        # Check for metabolically labile groups
        labile_patterns = [
            '[#6]O[#6]',           # Ethers (can be demethylated)
            '[#6]N([#6])[#6]',     # Tertiary amines (N-dealkylation)
            'c1ccc(O)cc1',         # Phenols (glucuronidation)
            '[#6]C(=O)O[#6]',      # Esters (hydrolysis)
        ]
        
        for pattern in labile_patterns:
            matches = len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
            score *= (1.0 - 0.1 * matches)  # 10% penalty per labile group
        
        # Bonus for halogen substitution (often more stable)
        halogens = ['F', 'Cl', 'Br', 'I']
        halogen_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in halogens)
        if halogen_count > 0:
            score *= 1.1  # 10% bonus
        
        return min(1.0, max(0.0, score))
    
    def _calculate_safety_score(self, mol: Chem.Mol) -> float:
        """Calculate safety score based on structural alerts."""
        score = 1.0
        
        # Structural alerts (PAINS, toxic groups, etc.)
        toxic_patterns = [
            'c1ccc2c(c1)oc1ccccc12',  # Dibenzofuran
            '[#6]=[#7+]=[#7-]',       # Diazo compounds
            '[#6]S(=O)(=O)N',         # Sulfonamides (potential for allergic reactions)
            'c1ccc(cc1)N=Nc2ccccc2',  # Azo compounds
        ]
        
        for pattern in toxic_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                score *= 0.7  # 30% penalty for toxic alerts
        
        # Check for high reactivity
        mw = Descriptors.MolWt(mol)
        if mw > 600:  # Very large molecules
            score *= 0.8
        
        # Check for excessive lipophilicity (off-target effects)
        logp = Crippen.MolLogP(mol)
        if logp > 6:
            score *= 0.6
        
        return max(0.0, score)
    
    def _calculate_psychedelic_profile(self, mol: Chem.Mol) -> float:
        """Score based on psychedelic-specific structural features."""
        score = 0.0
        
        # Check for psychedelic scaffolds
        psychedelic_patterns = [
            'c1ccc(cc1)CCN',          # Phenethylamine (2C series)
            'c1ccc(cc1)CC(C)N',       # Amphetamine (DOx series)
            'c1cc(O)c(OC)c(OC)c1',    # Mescaline-type
            'c1ccc2[nH]c3ccccc3c2c1', # Tryptamine core
        ]
        
        for pattern in psychedelic_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                score += 0.3
        
        # Halogen substitution bonus (important for 2C series)
        halogens = ['Br', 'I', 'Cl', 'F']
        halogen_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in halogens)
        if halogen_count == 2:  # Optimal for 2C series
            score += 0.4
        elif halogen_count == 1:
            score += 0.2
        
        # Aromatic ring systems
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        if aromatic_rings >= 1:
            score += 0.3
        
        return min(1.0, score)
    
    def _generate_optimization_recommendations(self, mol: Chem.Mol, scores: Dict) -> List[str]:
        """Generate specific optimization recommendations."""
        recommendations = []
        
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        
        # Molecular weight recommendations
        if mw > 450:
            recommendations.append("Reduce molecular weight for better BBB penetration")
        elif mw < 180:
            recommendations.append("Increase molecular weight for improved stability")
        
        # LogP recommendations
        if logp > 5:
            recommendations.append("Reduce lipophilicity to minimize off-target effects")
        elif logp < 1.5:
            recommendations.append("Increase lipophilicity for better membrane permeation")
        
        # TPSA recommendations
        if tpsa > 70:
            recommendations.append("Reduce polar surface area for improved BBB penetration")
        elif tpsa < 20:
            recommendations.append("Add polar groups for better solubility")
        
        # H-bond donor recommendations
        if hbd > 2:
            recommendations.append("Reduce H-bond donors for better CNS penetration")
        
        # Specific psychedelic recommendations
        if scores.get('psychedelic_profile_score', 0) < 0.5:
            recommendations.append("Consider adding psychedelic-relevant structural features")
        
        # Safety recommendations
        if scores.get('safety_score', 1) < 0.7:
            recommendations.append("Address potential safety concerns in structure")
        
        return recommendations
    
    def _classify_cns_potential(self, composite_score: float) -> str:
        """Classify CNS potential based on composite score."""
        if composite_score >= 0.8:
            return "Excellent CNS Potential"
        elif composite_score >= 0.6:
            return "Good CNS Potential"
        elif composite_score >= 0.4:
            return "Moderate CNS Potential"
        elif composite_score >= 0.2:
            return "Poor CNS Potential"
        else:
            return "Very Poor CNS Potential"

def optimize_compound_batch(compounds_df: pd.DataFrame) -> pd.DataFrame:
    """Batch CNS optimization analysis."""
    optimizer = CNSOptimizer()
    results = []
    
    for idx, row in compounds_df.iterrows():
        smiles = row['smiles']
        name = row.get('name', f'Compound_{idx}')
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                cns_analysis = optimizer.calculate_cns_score(mol)
                
                result = {
                    'name': name,
                    'smiles': smiles,
                    **cns_analysis,
                    'recommendations': ' | '.join(cns_analysis.get('optimization_recommendations', []))
                }
                
                # Remove the list from the result dict to avoid issues
                result.pop('optimization_recommendations', None)
                results.append(result)
                
        except Exception as e:
            print(f"Error analyzing {name}: {str(e)}")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Test with 2C-B
    optimizer = CNSOptimizer()
    test_mol = Chem.MolFromSmiles("CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN")
    
    print("🧠 Testing CNS Optimization Tools...")
    results = optimizer.calculate_cns_score(test_mol)
    
    print(f"\n📊 2C-B CNS Optimization Results:")
    print(f"Composite CNS Score: {results['composite_cns_score']:.3f}")
    print(f"CNS Classification: {results['cns_classification']}")
    print(f"BBB Score: {results['bbb_score']:.3f}")
    print(f"CNS MPO Score: {results['cns_mpo_score']:.3f}")
    print(f"Safety Score: {results['safety_score']:.3f}")
    
    print(f"\n💡 Optimization Recommendations:")
    for rec in results['optimization_recommendations']:
        print(f"   • {rec}")
    
    print("\n✅ CNS Optimization Tools Ready!")