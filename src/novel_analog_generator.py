"""
Novel Analog Generation for Psychedelic Therapeutics
====================================================

AI-powered molecular design for generating novel 2C-B derivatives with optimized properties:
- SMILES-based molecular generation
- Property-guided optimization
- Novel scaffold hopping
- Lead optimization workflows

Author: AI Assistant for CNS Therapeutics Research
Focus: AI-driven psychedelic drug design and novel analog generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Crippen
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not available. Limited molecular generation capabilities.")

# Import our custom modules
try:
    from .psychedelic_sar import TwoCBAnalyzer
    from .cns_optimization import CNSOptimizer
    from .ht2a_binding import HT2AReceptorPredictor
    from .predictive_models import PsychedelicMLPredictor
except ImportError:
    print("⚠️ Some analysis modules not available. Using simplified predictions.")

class NovelAnalogGenerator:
    """AI-powered generator for novel psychedelic analogs."""
    
    def __init__(self):
        self.output_dir = Path("generated_compounds")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize analysis modules
        if RDKIT_AVAILABLE:
            try:
                self.sar_analyzer = TwoCBAnalyzer()
                self.cns_optimizer = CNSOptimizer()
                self.ht2a_predictor = HT2AReceptorPredictor()
                self.ml_predictor = PsychedelicMLPredictor()
                self.analysis_available = True
            except:
                self.analysis_available = False
        else:
            self.analysis_available = False
        
        # 2C-B reference structure
        self.reference_2cb = "CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN"
        
        # Molecular building blocks for generation
        self.building_blocks = {
            'halogens': ['Br', 'Cl', 'I', 'F'],
            'alkyl_chains': ['C', 'CC', 'CCC', 'CC(C)', 'CCCC'],
            'aromatic_substitutions': ['OC', 'OCC', 'OCCC', 'N(C)C', 'CF3'],
            'linkers': ['O', 'S', 'NH', 'CH2', 'C(=O)'],
            'terminal_groups': ['N', 'NC', 'NCC', 'N(C)C', 'NCCC']
        }
        
        # Optimization targets
        self.optimization_targets = {
            'high_potency': {'ht2a_affinity': 8.0, 'cns_penetration': 0.8},
            'balanced_profile': {'ht2a_affinity': 7.0, 'cns_penetration': 0.7, 'safety': 0.8},
            'selective_binding': {'ht2a_affinity': 7.5, 'selectivity': 0.9},
            'oral_bioavailability': {'drug_likeness': 0.9, 'metabolic_stability': 0.8}
        }
    
    def generate_2cb_analogs(self, num_analogs: int = 50, target_profile: str = 'balanced_profile') -> pd.DataFrame:
        """
        Generate novel 2C-B analogs using systematic modification.
        
        Args:
            num_analogs: Number of analogs to generate
            target_profile: Optimization target ('high_potency', 'balanced_profile', etc.)
            
        Returns:
            DataFrame with generated analogs and predicted properties
        """
        print(f"🧬 Generating {num_analogs} novel 2C-B analogs...")
        
        generated_compounds = []
        
        # Strategy 1: Systematic halogen modifications
        halogen_analogs = self._generate_halogen_analogs()
        generated_compounds.extend(halogen_analogs[:num_analogs//4])
        
        # Strategy 2: Alkyl chain modifications
        alkyl_analogs = self._generate_alkyl_analogs()
        generated_compounds.extend(alkyl_analogs[:num_analogs//4])
        
        # Strategy 3: Aromatic ring modifications
        aromatic_analogs = self._generate_aromatic_analogs()
        generated_compounds.extend(aromatic_analogs[:num_analogs//4])
        
        # Strategy 4: Novel scaffold variations
        scaffold_analogs = self._generate_scaffold_variants()
        generated_compounds.extend(scaffold_analogs[:num_analogs//4])
        
        # Ensure we have enough compounds
        while len(generated_compounds) < num_analogs:
            random_analog = self._generate_random_analog()
            if random_analog:
                generated_compounds.append(random_analog)
        
        # Convert to DataFrame
        df = pd.DataFrame(generated_compounds[:num_analogs])
        
        # Add predicted properties
        if self.analysis_available:
            df = self._add_predicted_properties(df)
            df = self._score_compounds(df, target_profile)
        
        # Sort by optimization score
        if 'optimization_score' in df.columns:
            df = df.sort_values('optimization_score', ascending=False)
        
        print(f"✅ Generated {len(df)} novel analogs")
        return df
    
    def _generate_halogen_analogs(self) -> List[Dict]:
        """Generate analogs with different halogen substitutions."""
        analogs = []
        base_structure = "CCc1cc({})c(OCc2ccccc2)c({})c1CCN"
        
        # Single halogens
        for hal in self.building_blocks['halogens']:
            smiles = base_structure.format(hal, hal)
            analogs.append({
                'name': f'2C-{hal}',
                'smiles': smiles,
                'modification_type': 'halogen_substitution',
                'description': f'2C-B analog with {hal} substitution'
            })
        
        # Mixed halogens
        for hal1, hal2 in itertools.combinations(self.building_blocks['halogens'], 2):
            smiles = base_structure.format(hal1, hal2)
            analogs.append({
                'name': f'2C-{hal1}-{hal2}',
                'smiles': smiles,
                'modification_type': 'mixed_halogen',
                'description': f'2C-B analog with {hal1} and {hal2} substitution'
            })
        
        return analogs
    
    def _generate_alkyl_analogs(self) -> List[Dict]:
        """Generate analogs with different alkyl chain modifications."""
        analogs = []
        
        # Modify the ethyl chain
        for chain in self.building_blocks['alkyl_chains']:
            if chain != 'CC':  # Skip the original ethyl
                smiles = f"{chain}c1cc(Br)c(OCc2ccccc2)c(Br)c1CCN"
                if self._is_valid_smiles(smiles):
                    analogs.append({
                        'name': f'2C-B-{chain}',
                        'smiles': smiles,
                        'modification_type': 'alkyl_chain',
                        'description': f'2C-B analog with {chain} alkyl chain'
                    })
        
        # Modify the amine alkyl chain
        for chain in ['C', 'CC', 'CCC']:
            smiles = f"CCc1cc(Br)c(OCc2ccccc2)c(Br)c1{chain}N"
            if self._is_valid_smiles(smiles):
                analogs.append({
                    'name': f'2C-B-N{chain}',
                    'smiles': smiles,
                    'modification_type': 'amine_chain',
                    'description': f'2C-B analog with modified amine chain'
                })
        
        return analogs
    
    def _generate_aromatic_analogs(self) -> List[Dict]:
        """Generate analogs with aromatic ring modifications."""
        analogs = []
        
        # Modify the 4-position (currently methoxy in original structure)
        base_structure = "CCc1cc(Br)c({})c(Br)c1CCN"
        
        for substitution in self.building_blocks['aromatic_substitutions']:
            smiles = base_structure.format(substitution)
            if self._is_valid_smiles(smiles):
                analogs.append({
                    'name': f'2C-B-4-{substitution}',
                    'smiles': smiles,
                    'modification_type': 'aromatic_substitution',
                    'description': f'2C-B analog with 4-position {substitution} substitution'
                })
        
        # Additional ring systems
        bicyclic_base = "CCc1cc(Br)c2c(c1CCN)OCO2"  # Methylenedioxyphenyl
        if self._is_valid_smiles(bicyclic_base):
            analogs.append({
                'name': '2C-B-MDO',
                'smiles': bicyclic_base,
                'modification_type': 'ring_fusion',
                'description': '2C-B analog with methylenedioxy ring'
            })
        
        return analogs
    
    def _generate_scaffold_variants(self) -> List[Dict]:
        """Generate variants with modified core scaffolds."""
        analogs = []
        
        # DOx-type (amphetamine) variants
        dox_structure = "CC(N)Cc1cc(Br)c(OCc2ccccc2)c(Br)c1"
        analogs.append({
            'name': 'DOB-analog',
            'smiles': dox_structure,
            'modification_type': 'scaffold_hop',
            'description': 'DOx-type scaffold with 2C-B substitution pattern'
        })
        
        # Benzofuran variants
        benzofuran_structure = "CCc1cc(Br)c2oc3ccccc3c2c1CCN"
        if self._is_valid_smiles(benzofuran_structure):
            analogs.append({
                'name': '2C-B-BF',
                'smiles': benzofuran_structure,
                'modification_type': 'scaffold_hop',
                'description': 'Benzofuran scaffold variant of 2C-B'
            })
        
        # Thiophene variants
        thiophene_structure = "CCc1cc(Br)sc1CCN"
        if self._is_valid_smiles(thiophene_structure):
            analogs.append({
                'name': '2C-B-T',
                'smiles': thiophene_structure,
                'modification_type': 'scaffold_hop',
                'description': 'Thiophene scaffold variant of 2C-B'
            })
        
        return analogs
    
    def _generate_random_analog(self) -> Optional[Dict]:
        """Generate a random analog using fragment recombination."""
        try:
            # Random combination of building blocks
            hal1 = random.choice(self.building_blocks['halogens'])
            hal2 = random.choice(self.building_blocks['halogens'])
            chain = random.choice(self.building_blocks['alkyl_chains'])
            aromatic = random.choice(self.building_blocks['aromatic_substitutions'])
            
            # Create random structure
            smiles = f"{chain}c1cc({hal1})c({aromatic})c({hal2})c1CCN"
            
            if self._is_valid_smiles(smiles):
                return {
                    'name': f'Random-{hal1}-{hal2}-{len(chain)}',
                    'smiles': smiles,
                    'modification_type': 'random_combination',
                    'description': 'Randomly generated analog'
                }
        except:
            pass
        
        return None
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES string is valid."""
        if not RDKIT_AVAILABLE:
            # Basic validation without RDKit
            return len(smiles) > 5 and 'c1' in smiles and 'N' in smiles
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def _add_predicted_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add predicted properties using analysis modules."""
        enhanced_df = df.copy()
        
        # Initialize property lists
        properties = {
            'mw': [], 'logp': [], 'tpsa': [], 'drug_likeness': [],
            'bbb_score': [], 'cns_score': [], 'ht2a_affinity': [],
            'safety_score': [], 'novelty_score': []
        }
        
        for _, row in df.iterrows():
            smiles = row['smiles']
            
            if not RDKIT_AVAILABLE:
                # Fallback predictions
                props = self._fallback_predictions(smiles)
            else:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        props = self._calculate_full_properties(mol, smiles)
                    else:
                        props = self._fallback_predictions(smiles)
                except:
                    props = self._fallback_predictions(smiles)
            
            # Add properties to lists
            for key in properties:
                properties[key].append(props.get(key, 0.0))
        
        # Add all properties to DataFrame
        for key, values in properties.items():
            enhanced_df[key] = values
        
        return enhanced_df
    
    def _calculate_full_properties(self, mol: Any, smiles: str) -> Dict[str, float]:
        """Calculate full property set using available modules."""
        props = {
            'mw': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
        }
        
        # Add analysis from custom modules if available
        try:
            if hasattr(self, 'cns_optimizer'):
                cns_results = self.cns_optimizer.calculate_cns_score(mol)
                props.update({
                    'bbb_score': cns_results.get('bbb_score', 0.5),
                    'cns_score': cns_results.get('composite_cns_score', 0.5),
                    'drug_likeness': cns_results.get('drug_likeness_score', 0.5),
                    'safety_score': cns_results.get('safety_score', 0.5)
                })
            
            if hasattr(self, 'ht2a_predictor'):
                ht2a_results = self.ht2a_predictor.predict_binding_affinity(smiles)
                props['ht2a_affinity'] = ht2a_results.get('predicted_pki', 6.0)
        except:
            # Fallback to simple calculations
            props.update({
                'bbb_score': min(1.0, max(0.0, (450 - props['mw']) / 300 + (4 - abs(props['logp'] - 3)) / 4)),
                'cns_score': min(1.0, max(0.0, (500 - props['mw']) / 400 + (3 - abs(props['logp'] - 2.5)) / 3)),
                'drug_likeness': 1.0 if props['mw'] < 500 and props['logp'] < 5 else 0.5,
                'ht2a_affinity': 6.0 + props['logp'] * 0.5 - props['mw'] / 100,
                'safety_score': 1.0 if props['mw'] < 400 and props['logp'] < 4 else 0.7
            })
        
        # Calculate novelty score
        props['novelty_score'] = self._calculate_novelty_score(smiles)
        
        return props
    
    def _fallback_predictions(self, smiles: str) -> Dict[str, float]:
        """Fallback property predictions without RDKit."""
        # Very basic estimations based on SMILES
        length = len(smiles)
        
        return {
            'mw': length * 12,  # Rough MW estimate
            'logp': smiles.count('c') * 0.5 + smiles.count('C') * 0.3 - smiles.count('N') * 0.5,
            'tpsa': smiles.count('O') * 20 + smiles.count('N') * 10,
            'drug_likeness': 0.7 if 20 < length < 60 else 0.3,
            'bbb_score': 0.6 if 25 < length < 50 else 0.3,
            'cns_score': 0.5,
            'ht2a_affinity': 6.0 + random.uniform(-1, 1),
            'safety_score': 0.7,
            'novelty_score': random.uniform(0.5, 1.0)
        }
    
    def _calculate_novelty_score(self, smiles: str) -> float:
        """Calculate novelty score compared to known compounds."""
        # Simple novelty scoring - in practice would use molecular fingerprints
        known_patterns = ['CCc1cc(Br)c(OC)', 'CC(N)Cc1cc', 'COc1cc(CCN)']
        
        novelty = 1.0
        for pattern in known_patterns:
            if pattern in smiles:
                novelty *= 0.8
        
        # Bonus for unique structural features
        unique_features = ['CF3', 'S(=O)', 'N(C)(C)', '[nH]']
        for feature in unique_features:
            if feature in smiles:
                novelty *= 1.1
        
        return min(1.0, novelty)
    
    def _score_compounds(self, df: pd.DataFrame, target_profile: str) -> pd.DataFrame:
        """Score compounds based on optimization target."""
        enhanced_df = df.copy()
        
        target = self.optimization_targets.get(target_profile, self.optimization_targets['balanced_profile'])
        
        scores = []
        for _, row in df.iterrows():
            score = 0.0
            
            # Score based on target criteria
            for criterion, target_value in target.items():
                if criterion in row:
                    actual_value = row[criterion]
                    # Score based on proximity to target
                    if target_value > 5:  # Assume this is affinity (higher is better)
                        score += min(1.0, actual_value / target_value)
                    else:  # Assume this is a ratio (closer to target is better)
                        score += max(0.0, 1.0 - abs(actual_value - target_value))
            
            # Normalize score
            scores.append(score / len(target))
        
        enhanced_df['optimization_score'] = scores
        
        return enhanced_df
    
    def export_compounds(self, df: pd.DataFrame, filename: str = "novel_2cb_analogs.csv") -> str:
        """Export generated compounds to file."""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        
        # Also create SDF file if RDKit available
        if RDKIT_AVAILABLE:
            sdf_path = self.output_dir / filename.replace('.csv', '.sdf')
            self._export_sdf(df, sdf_path)
        
        print(f"📁 Exported {len(df)} compounds to {output_path}")
        return str(output_path)
    
    def _export_sdf(self, df: pd.DataFrame, sdf_path: Path):
        """Export to SDF format with 3D coordinates."""
        writer = Chem.SDWriter(str(sdf_path))
        
        for _, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.UFFOptimizeMolecule(mol)
                    
                    # Add properties as molecule properties
                    for col in df.columns:
                        if col not in ['smiles', 'name']:
                            mol.SetProp(col, str(row[col]))
                    
                    mol.SetProp('_Name', row['name'])
                    writer.write(mol)
            except:
                continue
        
        writer.close()
    
    def generate_optimization_report(self, df: pd.DataFrame) -> str:
        """Generate comprehensive optimization report."""
        report_lines = [
            "# Novel 2C-B Analog Generation Report",
            f"Generated: {len(df)} compounds",
            "",
            "## Summary Statistics",
            f"Average Molecular Weight: {df['mw'].mean():.1f} ± {df['mw'].std():.1f}",
            f"Average LogP: {df['logp'].mean():.2f} ± {df['logp'].std():.2f}",
            f"Average BBB Score: {df['bbb_score'].mean():.2f} ± {df['bbb_score'].std():.2f}",
            f"Average 5-HT2A Affinity: {df['ht2a_affinity'].mean():.1f} ± {df['ht2a_affinity'].std():.1f}",
            "",
            "## Top 10 Compounds by Optimization Score",
        ]
        
        top_compounds = df.nlargest(10, 'optimization_score')
        for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
            report_lines.append(
                f"{i}. {row['name']} (Score: {row['optimization_score']:.3f})"
            )
            report_lines.append(f"   SMILES: {row['smiles']}")
            report_lines.append(f"   MW: {row['mw']:.1f}, LogP: {row['logp']:.2f}, 5-HT2A: {row['ht2a_affinity']:.1f}")
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / "generation_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return str(report_path)

def generate_psychedelic_library(num_compounds: int = 100, target_profile: str = 'balanced_profile') -> Dict[str, str]:
    """
    Main function to generate a comprehensive psychedelic compound library.
    
    Args:
        num_compounds: Number of compounds to generate
        target_profile: Optimization target profile
        
    Returns:
        Dictionary with file paths of generated outputs
    """
    print("🧬 Starting Novel Psychedelic Analog Generation...")
    
    generator = NovelAnalogGenerator()
    
    # Generate analogs
    analogs_df = generator.generate_2cb_analogs(num_compounds, target_profile)
    
    # Export results
    outputs = {}
    outputs['compounds_csv'] = generator.export_compounds(analogs_df)
    outputs['optimization_report'] = generator.generate_optimization_report(analogs_df)
    
    print(f"✅ Generated {len(analogs_df)} novel compounds")
    print(f"📊 Top compound: {analogs_df.iloc[0]['name']} (Score: {analogs_df.iloc[0]['optimization_score']:.3f})")
    
    return outputs

if __name__ == "__main__":
    # Test the generator
    print("🧬 Testing Novel Analog Generator...")
    
    generator = NovelAnalogGenerator()
    
    # Generate small test set
    test_analogs = generator.generate_2cb_analogs(10, 'balanced_profile')
    
    print(f"\n📊 Generated {len(test_analogs)} test analogs:")
    for i, (_, row) in enumerate(test_analogs.head().iterrows(), 1):
        print(f"{i}. {row['name']}: {row['smiles']}")
        if 'optimization_score' in row:
            print(f"   Score: {row['optimization_score']:.3f}")
    
    print("\n✅ Novel Analog Generator Ready!")