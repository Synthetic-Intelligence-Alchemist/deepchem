"""
Advanced Structure-Activity Relationship (SAR) Analysis for Psychedelic Compounds
=================================================================================

Specialized module for analyzing 2C-B and related psychedelic compounds with focus on:
- 5-HT2A receptor binding predictions
- CNS penetration optimization
- Psychedelic-specific molecular descriptors
- Advanced SAR correlation analysis

Author: AI Assistant for CNS Therapeutics Research
Target: 5-HT2A receptor-targeted psychedelic drug design
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# RDKit imports for advanced molecular analysis
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem import Fragments, GraphDescriptors, EState
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem import rdRGroupDecomposition as RGD

class PsychedelicSARAnalyzer:
    """Advanced SAR analysis specifically for psychedelic compounds."""
    
    def __init__(self):
        self.compound_classes = {
            '2C-series': ['2C-B', '2C-I', '2C-E', '2C-P', '2C-T-2', '2C-T-7', '2C-D', '2C-H'],
            'DOx-series': ['DOB', 'DOI', 'DOM', 'DOC', 'DON'],
            'Mescaline-analog': ['Mescaline', 'Escaline', 'Proscaline', 'Allylescaline'],
            'NBOMe-series': ['25B-NBOMe', '25I-NBOMe', '25C-NBOMe', '25D-NBOMe'],
            'Tryptamine': ['DMT', '5-MeO-DMT', 'DPT', 'DiPT'],
            'Lysergamide': ['LSD', 'ALD-52', '1P-LSD', 'ETH-LAD']
        }
        
        # 5-HT2A receptor specific structural features
        self.ht2a_pharmacophore = {
            'aromatic_rings': 1,  # Essential aromatic ring
            'amine_nitrogen': 1,  # Primary or secondary amine
            'halogen_substitution': ['Br', 'I', 'Cl', 'F'],  # Common in 2C series
            'methoxy_groups': ['OCH3', 'OC'],  # Mescaline-type substitutions
            'optimal_chain_length': 2  # C-C-N chain from aromatic ring
        }
    
    def calculate_psychedelic_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate specialized molecular descriptors relevant to psychedelic activity.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of psychedelic-specific descriptors
        """
        if mol is None:
            return {}
        
        try:
            descriptors = {
                # Basic physicochemical properties
                'mw': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                
                # CNS-relevant descriptors
                'cns_mpo': self._calculate_cns_mpo(mol),
                'bbb_score': self._calculate_bbb_score(mol),
                'lipinski_violations': self._count_lipinski_violations(mol),
                
                # Psychedelic-specific features
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'aliphatic_rings': Descriptors.NumAliphaticRings(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
                
                # Halogen analysis (important for 2C series)
                'halogen_count': self._count_halogens(mol),
                'bromine_count': self._count_element(mol, 'Br'),
                'iodine_count': self._count_element(mol, 'I'),
                'chlorine_count': self._count_element(mol, 'Cl'),
                'fluorine_count': self._count_element(mol, 'F'),
                
                # Electronic properties
                'kappa1': GraphDescriptors.Kappa1(mol),
                'kappa2': GraphDescriptors.Kappa2(mol),
                'kappa3': GraphDescriptors.Kappa3(mol),
                
                # Pharmacophore features
                'pharmacophore_score': self._calculate_pharmacophore_score(mol),
                'substitution_pattern': self._analyze_substitution_pattern(mol),
                
                # 5-HT2A binding prediction features
                'ht2a_affinity_pred': self._predict_ht2a_affinity(mol),
                'selectivity_index': self._calculate_selectivity_index(mol)
            }
            
            return descriptors
            
        except Exception as e:
            print(f"Error calculating descriptors: {str(e)}")
            return {}
    
    def _calculate_cns_mpo(self, mol: Chem.Mol) -> float:
        """Calculate CNS Multiparameter Optimization (MPO) score."""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # CNS MPO scoring (0-6 scale)
        mw_score = 1.0 if mw <= 360 else max(0, (460 - mw) / 100)
        logp_score = 1.0 if 1 <= logp <= 3 else max(0, 1 - abs(logp - 2) / 2)
        hbd_score = 1.0 if hbd <= 0.5 else max(0, (3 - hbd) / 2.5)
        tpsa_score = 1.0 if tpsa <= 40 else max(0, (90 - tpsa) / 50)
        
        cns_mpo = (mw_score + logp_score + hbd_score + tpsa_score) / 4 * 6
        return min(6.0, max(0.0, cns_mpo))
    
    def _calculate_bbb_score(self, mol: Chem.Mol) -> float:
        """Enhanced BBB penetration score for psychedelics."""
        tpsa = Descriptors.TPSA(mol)
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        
        # Optimized for psychedelic compounds
        tpsa_score = 1.0 if tpsa <= 60 else max(0, (90 - tpsa) / 30)
        logp_score = 1.0 if 2 <= logp <= 5 else max(0, 1 - abs(logp - 3.5) / 3)
        mw_score = 1.0 if mw <= 450 else max(0, (600 - mw) / 150)
        
        return (tpsa_score + logp_score + mw_score) / 3
    
    def _count_lipinski_violations(self, mol: Chem.Mol) -> int:
        """Count Lipinski Rule of Five violations."""
        violations = 0
        if Descriptors.MolWt(mol) > 500: violations += 1
        if Crippen.MolLogP(mol) > 5: violations += 1
        if Descriptors.NumHDonors(mol) > 5: violations += 1
        if Descriptors.NumHAcceptors(mol) > 10: violations += 1
        return violations
    
    def _count_halogens(self, mol: Chem.Mol) -> int:
        """Count total halogen atoms (important for 2C series)."""
        halogens = ['F', 'Cl', 'Br', 'I']
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in halogens:
                count += 1
        return count
    
    def _count_element(self, mol: Chem.Mol, element: str) -> int:
        """Count specific element in molecule."""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == element:
                count += 1
        return count
    
    def _calculate_pharmacophore_score(self, mol: Chem.Mol) -> float:
        """Score based on 5-HT2A pharmacophore features."""
        score = 0.0
        
        # Essential features for 5-HT2A activity
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        if aromatic_rings >= 1:
            score += 2.0
        
        # Amine nitrogen (basic center)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0:
                score += 2.0
                break
        
        # Halogen substitution (2C series)
        if self._count_halogens(mol) > 0:
            score += 1.0
        
        # Methoxy groups (mescaline-type)
        methoxy_pattern = Chem.MolFromSmarts('[#6][#8][#6]')
        if mol.HasSubstructMatch(methoxy_pattern):
            score += 1.0
        
        return min(6.0, score)
    
    def _analyze_substitution_pattern(self, mol: Chem.Mol) -> float:
        """Analyze substitution pattern on aromatic ring."""
        # Look for common psychedelic substitution patterns
        # 2,5-disubstitution (2C series), 2,4,5-trisubstitution (DOx), etc.
        
        # This is a simplified analysis - could be expanded with SMARTS patterns
        aromatic_carbons = [atom for atom in mol.GetAtoms() 
                          if atom.GetIsAromatic() and atom.GetSymbol() == 'C']
        
        substituted_positions = 0
        for atom in aromatic_carbons:
            if atom.GetDegree() > 2:  # Substituted position
                substituted_positions += 1
        
        # Score based on typical psychedelic substitution patterns
        if substituted_positions in [2, 3]:  # Optimal for most psychedelics
            return 1.0
        elif substituted_positions in [1, 4]:
            return 0.7
        else:
            return 0.3
    
    def _predict_ht2a_affinity(self, mol: Chem.Mol) -> float:
        """
        Predict 5-HT2A receptor affinity based on known SAR.
        This is a simplified model - in practice, you'd use trained ML models.
        """
        base_score = 5.0  # Baseline pKi
        
        # Molecular weight penalty
        mw = Descriptors.MolWt(mol)
        if mw > 400:
            base_score -= (mw - 400) / 100
        
        # LogP optimization
        logp = Crippen.MolLogP(mol)
        if 2 <= logp <= 5:
            base_score += 1.0
        elif logp > 6:
            base_score -= 0.5
        
        # Halogen bonus (2C series)
        halogen_count = self._count_halogens(mol)
        if halogen_count == 2:  # Optimal for 2C series
            base_score += 1.5
        elif halogen_count == 1:
            base_score += 0.5
        
        # Pharmacophore bonus
        base_score += self._calculate_pharmacophore_score(mol) / 6 * 2
        
        return max(0.0, min(10.0, base_score))
    
    def _calculate_selectivity_index(self, mol: Chem.Mol) -> float:
        """Calculate selectivity index for 5-HT2A vs other receptors."""
        # Simplified selectivity prediction
        # In practice, this would involve multiple receptor models
        
        tpsa = Descriptors.TPSA(mol)
        logp = Crippen.MolLogP(mol)
        
        # Compounds with moderate TPSA and LogP tend to be more selective
        selectivity = 1.0
        if tpsa < 30 or tpsa > 80:
            selectivity -= 0.3
        if logp < 1 or logp > 6:
            selectivity -= 0.3
        
        return max(0.0, selectivity)

class TwoCBAnalyzer:
    """Specialized analyzer for 2C-B and its analogs."""
    
    def __init__(self):
        self.sar_analyzer = PsychedelicSARAnalyzer()
        
        # Known 2C-B structure (reference compound)
        self.reference_2cb = "CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN"
        
        # Structural modification sites
        self.modification_sites = {
            'N-position': 'Primary amine can be modified',
            'alpha-position': 'Carbon alpha to amine',
            'beta-position': 'Carbon beta to amine (ethyl group)',
            '2-position': 'Aromatic substitution',
            '5-position': 'Aromatic substitution',
            '4-position': 'Methoxy or other substituent'
        }
    
    def analyze_2cb_analog(self, smiles: str, name: str = None) -> Dict:
        """
        Comprehensive analysis of 2C-B analog.
        
        Args:
            smiles: SMILES string of the analog
            name: Optional name of the compound
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}
        
        # Calculate all descriptors
        descriptors = self.sar_analyzer.calculate_psychedelic_descriptors(mol)
        
        # Compare to 2C-B reference
        reference_mol = Chem.MolFromSmiles(self.reference_2cb)
        reference_descriptors = self.sar_analyzer.calculate_psychedelic_descriptors(reference_mol)
        
        analysis = {
            'compound_name': name or 'Unknown',
            'smiles': smiles,
            'descriptors': descriptors,
            'comparison_to_2cb': self._compare_to_reference(descriptors, reference_descriptors),
            'structural_classification': self._classify_structure(mol),
            'sar_insights': self._generate_sar_insights(descriptors),
            'optimization_suggestions': self._suggest_optimizations(descriptors),
            'safety_flags': self._identify_safety_concerns(descriptors),
            'synthesis_complexity': self._estimate_synthesis_complexity(mol)
        }
        
        return analysis
    
    def _compare_to_reference(self, descriptors: Dict, reference: Dict) -> Dict:
        """Compare compound to 2C-B reference."""
        comparison = {}
        
        key_properties = ['mw', 'logp', 'tpsa', 'cns_mpo', 'ht2a_affinity_pred']
        
        for prop in key_properties:
            if prop in descriptors and prop in reference:
                diff = descriptors[prop] - reference[prop]
                percent_diff = (diff / reference[prop]) * 100 if reference[prop] != 0 else 0
                
                comparison[prop] = {
                    'value': descriptors[prop],
                    'reference': reference[prop],
                    'difference': diff,
                    'percent_change': percent_diff,
                    'interpretation': self._interpret_change(prop, percent_diff)
                }
        
        return comparison
    
    def _classify_structure(self, mol: Chem.Mol) -> Dict:
        """Classify the structural type and modifications."""
        classification = {
            'scaffold': 'Unknown',
            'substitution_pattern': 'Unknown',
            'modifications': []
        }
        
        # Check for 2C scaffold
        phenethylamine_pattern = Chem.MolFromSmarts('c1ccccc1CCN')
        if mol.HasSubstructMatch(phenethylamine_pattern):
            classification['scaffold'] = '2C-type (phenethylamine)'
        
        # Check for DOx scaffold
        amphetamine_pattern = Chem.MolFromSmarts('c1ccccc1CC(C)N')
        if mol.HasSubstructMatch(amphetamine_pattern):
            classification['scaffold'] = 'DOx-type (amphetamine)'
        
        # Analyze substitutions
        halogen_count = sum([
            self.sar_analyzer._count_element(mol, 'Br'),
            self.sar_analyzer._count_element(mol, 'I'),
            self.sar_analyzer._count_element(mol, 'Cl'),
            self.sar_analyzer._count_element(mol, 'F')
        ])
        
        if halogen_count == 2:
            classification['substitution_pattern'] = '2,5-dihalogen (classic 2C)'
        elif halogen_count == 1:
            classification['substitution_pattern'] = 'monohalogen'
        
        return classification
    
    def _generate_sar_insights(self, descriptors: Dict) -> List[str]:
        """Generate structure-activity relationship insights."""
        insights = []
        
        # CNS penetration insights
        if descriptors.get('bbb_score', 0) > 0.8:
            insights.append("Excellent BBB penetration predicted - suitable for CNS activity")
        elif descriptors.get('bbb_score', 0) < 0.5:
            insights.append("Poor BBB penetration - may require structural modifications")
        
        # Potency predictions
        if descriptors.get('ht2a_affinity_pred', 0) > 7:
            insights.append("High predicted 5-HT2A affinity - likely potent psychedelic")
        elif descriptors.get('ht2a_affinity_pred', 0) < 5:
            insights.append("Low predicted 5-HT2A affinity - may be inactive")
        
        # Drug-likeness
        if descriptors.get('lipinski_violations', 0) == 0:
            insights.append("No Lipinski violations - good drug-like properties")
        
        # Selectivity
        if descriptors.get('selectivity_index', 0) > 0.8:
            insights.append("High selectivity index - likely selective for 5-HT2A")
        
        return insights
    
    def _suggest_optimizations(self, descriptors: Dict) -> List[str]:
        """Suggest structural optimizations."""
        suggestions = []
        
        # Molecular weight optimization
        if descriptors.get('mw', 0) > 450:
            suggestions.append("Consider reducing molecular weight to improve CNS penetration")
        
        # LogP optimization
        logp = descriptors.get('logp', 0)
        if logp > 5:
            suggestions.append("Consider adding polar groups to reduce lipophilicity")
        elif logp < 2:
            suggestions.append("Consider adding lipophilic groups to improve membrane permeation")
        
        # TPSA optimization
        if descriptors.get('tpsa', 0) > 70:
            suggestions.append("Reduce polar surface area to improve BBB penetration")
        
        # Halogen optimization for 2C series
        if descriptors.get('halogen_count', 0) == 0:
            suggestions.append("Consider adding halogens at 2,5-positions for 2C-type activity")
        
        return suggestions
    
    def _identify_safety_concerns(self, descriptors: Dict) -> List[str]:
        """Identify potential safety concerns."""
        concerns = []
        
        # High lipophilicity concerns
        if descriptors.get('logp', 0) > 6:
            concerns.append("Very high lipophilicity - risk of off-target effects")
        
        # Molecular weight concerns
        if descriptors.get('mw', 0) > 500:
            concerns.append("High molecular weight - potential for drug-drug interactions")
        
        # Halogen concerns
        if descriptors.get('iodine_count', 0) > 0:
            concerns.append("Iodine substitution - potential thyroid effects")
        
        return concerns
    
    def _estimate_synthesis_complexity(self, mol: Chem.Mol) -> Dict:
        """Estimate synthetic accessibility."""
        # Simplified synthesis complexity estimation
        complexity_score = 1.0
        
        # Ring complexity
        rings = Descriptors.RingCount(mol)
        complexity_score += rings * 0.5
        
        # Halogen substitutions (may require special conditions)
        halogens = self.sar_analyzer._count_halogens(mol)
        complexity_score += halogens * 0.3
        
        # Stereochemistry (not captured in SMILES for this analysis)
        # Would need 3D structure analysis
        
        if complexity_score <= 2:
            difficulty = "Easy"
        elif complexity_score <= 4:
            difficulty = "Moderate"
        else:
            difficulty = "Difficult"
        
        return {
            'complexity_score': complexity_score,
            'difficulty': difficulty,
            'estimated_steps': max(3, int(complexity_score * 2))
        }
    
    def _interpret_change(self, property_name: str, percent_change: float) -> str:
        """Interpret the significance of property changes."""
        abs_change = abs(percent_change)
        
        if abs_change < 5:
            return "Minimal change"
        elif abs_change < 20:
            return "Moderate change"
        elif abs_change < 50:
            return "Significant change"
        else:
            return "Major change"

def batch_analyze_2cb_analogs(compounds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch analysis of 2C-B analogs from a DataFrame.
    
    Args:
        compounds_df: DataFrame with 'smiles' and 'name' columns
        
    Returns:
        Enhanced DataFrame with SAR analysis
    """
    analyzer = TwoCBAnalyzer()
    results = []
    
    for idx, row in compounds_df.iterrows():
        smiles = row['smiles']
        name = row.get('name', f'Compound_{idx}')
        
        try:
            analysis = analyzer.analyze_2cb_analog(smiles, name)
            
            # Flatten the results for DataFrame
            flat_result = {
                'name': name,
                'smiles': smiles,
                **analysis['descriptors'],
                'structural_scaffold': analysis['structural_classification']['scaffold'],
                'substitution_pattern': analysis['structural_classification']['substitution_pattern'],
                'sar_insights': ' | '.join(analysis['sar_insights']),
                'optimization_suggestions': ' | '.join(analysis['optimization_suggestions']),
                'safety_concerns': ' | '.join(analysis['safety_flags']),
                'synthesis_difficulty': analysis['synthesis_complexity']['difficulty']
            }
            
            results.append(flat_result)
            
        except Exception as e:
            print(f"Error analyzing {name}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Test the analyzer with 2C-B
    analyzer = TwoCBAnalyzer()
    
    # Test compound: 2C-B
    test_smiles = "CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN"
    
    print("🧪 Testing 2C-B SAR Analyzer...")
    analysis = analyzer.analyze_2cb_analog(test_smiles, "2C-B")
    
    print(f"\n📊 Analysis Results for 2C-B:")
    print(f"CNS MPO Score: {analysis['descriptors']['cns_mpo']:.2f}/6")
    print(f"BBB Score: {analysis['descriptors']['bbb_score']:.2f}")
    print(f"Predicted 5-HT2A Affinity: {analysis['descriptors']['ht2a_affinity_pred']:.2f}")
    print(f"Pharmacophore Score: {analysis['descriptors']['pharmacophore_score']:.2f}")
    
    print(f"\n🔍 SAR Insights:")
    for insight in analysis['sar_insights']:
        print(f"   • {insight}")
    
    print(f"\n⚗️ Synthesis Complexity: {analysis['synthesis_complexity']['difficulty']}")
    print("\n✅ 2C-B SAR Analysis Module Ready!")