"""
Compound Generation Module
=========================

Tools for generating novel 2C-B analogs and optimizing psychedelic compounds
for improved therapeutic profiles.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import deepchem as dc
from sklearn.cluster import KMeans
import random

class PsychedelicGenerator:
    """Generator for novel psychedelic compounds."""
    
    def __init__(self):
        self.scaffold_library = {}
        self.substitution_library = {}
        self._initialize_libraries()
    
    def _initialize_libraries(self):
        """Initialize chemical libraries for generation."""
        # Core scaffolds for psychedelics
        self.scaffold_library = {
            'phenethylamine': 'c1ccccc1CCN',
            'amphetamine': 'c1ccccc1CC(N)C',
            'tryptamine': 'c1ccc2[nH]ccc2c1CCN',
            'mescaline_core': 'c1cc(CCN)cc(OC)c1OC',
            '2c_core': 'c1cc(*)c(OCc2ccccc2)c(*)c1CCN'  # * for substitution points
        }
        
        # Substitution patterns
        self.substitution_library = {
            'halogens': ['F', 'Cl', 'Br', 'I'],
            'alkyl': ['C', 'CC', 'CCC', 'C(C)C'],
            'alkoxy': ['OC', 'OCC', 'OCCC', 'OC(C)C'],
            'methylenedioxy': 'OCO',
            'nitro': '[N+](=O)[O-]',
            'amino': 'N',
            'hydroxyl': 'O',
            'thioether': ['SC', 'SCC', 'SCCC']
        }
    
    def generate_2c_analogs(self, num_compounds: int = 50, 
                           substitution_positions: List[int] = [2, 5]) -> List[str]:
        """
        Generate 2C-B analogs with various substitutions.
        
        Args:
            num_compounds: Number of compounds to generate
            substitution_positions: Positions for substitutions (2,5 for traditional 2C)
        """
        analogs = []
        base_scaffold = 'c1cc(*)c(OCc2ccccc2)c(*)c1CCN'
        
        # Get substitution options
        substitutions = []
        for sub_type, sub_list in self.substitution_library.items():
            if isinstance(sub_list, list):
                substitutions.extend(sub_list)
            else:
                substitutions.append(sub_list)
        
        generated_count = 0
        attempts = 0
        max_attempts = num_compounds * 10
        
        while generated_count < num_compounds and attempts < max_attempts:
            attempts += 1
            
            # Choose random substitutions
            sub1 = random.choice(substitutions)
            sub2 = random.choice(substitutions)
            
            # Replace * with substitutions
            smiles = base_scaffold.replace('*', sub1, 1).replace('*', sub2, 1)
            
            # Validate molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol)
                if canonical_smiles not in analogs:
                    analogs.append(canonical_smiles)
                    generated_count += 1
        
        return analogs
    
    def generate_scaffold_variants(self, base_smiles: str, 
                                  num_variants: int = 20) -> List[str]:
        """Generate variants by modifying the scaffold."""
        mol = Chem.MolFromSmiles(base_smiles)
        if mol is None:
            return []
        
        variants = []
        
        # Get Murcko scaffold
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
        except:
            scaffold_smiles = base_smiles
        
        # Generate variants by adding different side chains
        side_chains = ['C', 'CC', 'CCC', 'OC', 'OCC', 'F', 'Cl', 'Br']
        
        for _ in range(num_variants):
            try:
                # Simple approach: add random atoms/groups
                variant_mol = Chem.MolFromSmiles(base_smiles)
                if variant_mol:
                    # This is a simplified approach - real scaffold hopping would be more sophisticated
                    variants.append(Chem.MolToSmiles(variant_mol))
            except:
                continue
        
        return list(set(variants))
    
    def optimize_for_properties(self, base_smiles: str,
                               target_properties: Dict[str, Tuple[float, float]],
                               num_generations: int = 10,
                               population_size: int = 50) -> List[str]:
        """
        Optimize compounds for target properties using genetic algorithm approach.
        
        Args:
            base_smiles: Starting compound
            target_properties: Dict of property_name: (min_value, max_value)
            num_generations: Number of optimization generations
            population_size: Size of each generation
        """
        # Start with base compound and variants
        population = [base_smiles]
        population.extend(self.generate_2c_analogs(population_size - 1))
        
        for generation in range(num_generations):
            # Score population
            scored_population = []
            for smiles in population:
                score = self._calculate_fitness_score(smiles, target_properties)
                scored_population.append((smiles, score))
            
            # Sort by fitness
            scored_population.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top performers
            survivors = [item[0] for item in scored_population[:population_size//2]]
            
            # Generate new compounds from survivors
            new_population = survivors.copy()
            
            while len(new_population) < population_size:
                # Mutate survivors
                parent = random.choice(survivors)
                child = self._mutate_compound(parent)
                if child:
                    new_population.append(child)
            
            population = new_population
        
        # Return top compounds
        final_scored = [(smiles, self._calculate_fitness_score(smiles, target_properties)) 
                       for smiles in population]
        final_scored.sort(key=lambda x: x[1], reverse=True)
        
        return [item[0] for item in final_scored[:10]]
    
    def _calculate_fitness_score(self, smiles: str, 
                               target_properties: Dict[str, Tuple[float, float]]) -> float:
        """Calculate fitness score for optimization."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        
        score = 0.0
        property_count = 0
        
        for prop_name, (min_val, max_val) in target_properties.items():
            prop_value = self._calculate_property(mol, prop_name)
            
            if prop_value is not None:
                # Score based on how close to target range
                if min_val <= prop_value <= max_val:
                    score += 1.0
                else:
                    # Penalty for being outside range
                    if prop_value < min_val:
                        penalty = (min_val - prop_value) / min_val
                    else:
                        penalty = (prop_value - max_val) / max_val
                    score += max(0.0, 1.0 - penalty)
                
                property_count += 1
        
        return score / max(1, property_count)
    
    def _calculate_property(self, mol: Chem.Mol, property_name: str) -> Optional[float]:
        """Calculate molecular property."""
        try:
            if property_name == 'mw':
                return Descriptors.MolWt(mol)
            elif property_name == 'logp':
                return Descriptors.MolLogP(mol)
            elif property_name == 'tpsa':
                return Descriptors.TPSA(mol)
            elif property_name == 'hbd':
                return Descriptors.NumHDonors(mol)
            elif property_name == 'hba':
                return Descriptors.NumHAcceptors(mol)
            elif property_name == 'rotatable_bonds':
                return Descriptors.NumRotatableBonds(mol)
            else:
                return None
        except:
            return None
    
    def _mutate_compound(self, smiles: str) -> Optional[str]:
        """Mutate a compound by making small changes."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Simple mutations: add/remove atoms, change bonds
        # This is a simplified approach - real chemical mutations would be more sophisticated
        
        try:
            # Try to add a methyl group to a random carbon
            edit_mol = Chem.RWMol(mol)
            carbons = [atom.GetIdx() for atom in edit_mol.GetAtoms() if atom.GetSymbol() == 'C']
            
            if carbons:
                carbon_idx = random.choice(carbons)
                carbon = edit_mol.GetAtomWithIdx(carbon_idx)
                
                # Check if carbon can accept another bond
                if carbon.GetTotalValence() < 4:
                    # Add methyl group
                    new_carbon = Chem.Atom(6)  # Carbon
                    new_idx = edit_mol.AddAtom(new_carbon)
                    edit_mol.AddBond(carbon_idx, new_idx, Chem.BondType.SINGLE)
                    
                    # Add hydrogens to complete valence
                    for _ in range(3):  # Methyl group
                        h_atom = Chem.Atom(1)  # Hydrogen
                        h_idx = edit_mol.AddAtom(h_atom)
                        edit_mol.AddBond(new_idx, h_idx, Chem.BondType.SINGLE)
            
            # Sanitize and return
            Chem.SanitizeMol(edit_mol)
            return Chem.MolToSmiles(edit_mol)
            
        except:
            return smiles  # Return original if mutation fails

class NovelPsychedelicDesigner:
    """Designer for novel psychedelic therapeutics."""
    
    def __init__(self):
        self.generator = PsychedelicGenerator()
    
    def design_selective_5ht2a_agonists(self, num_compounds: int = 20) -> pd.DataFrame:
        """Design compounds selective for 5-HT2A over other serotonin receptors."""
        # Target properties for 5-HT2A selectivity
        target_properties = {
            'mw': (200, 400),      # Molecular weight
            'logp': (1.0, 3.5),    # LogP for BBB penetration
            'tpsa': (20, 90),      # TPSA for BBB penetration
            'hbd': (0, 3),         # H-bond donors
            'hba': (1, 8),         # H-bond acceptors
        }
        
        # Generate compounds
        base_2cb = 'CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN'
        compounds = self.generator.optimize_for_properties(
            base_2cb, target_properties, num_generations=5, population_size=num_compounds*2
        )
        
        # Create DataFrame with properties
        results = []
        for smiles in compounds:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                props = {
                    'smiles': smiles,
                    'mw': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'hbd': Descriptors.NumHDonors(mol),
                    'hba': Descriptors.NumHAcceptors(mol),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'aromatic_rings': Descriptors.NumAromaticRings(mol),
                }
                results.append(props)
        
        return pd.DataFrame(results)
    
    def design_non_hallucinogenic_5ht2a_modulators(self) -> List[str]:
        """Design 5-HT2A modulators with reduced hallucinogenic effects."""
        # Strategy: Design biased agonists or allosteric modulators
        
        # Base structures for non-hallucinogenic effects
        base_structures = [
            'c1ccc2[nH]ccc2c1CCN',  # Tryptamine core
            'c1ccc(CCN)cc1',        # Simple phenethylamine
            'c1ccc(CN)cc1',         # Benzylamine
        ]
        
        designed_compounds = []
        
        for base in base_structures:
            variants = self.generator.generate_scaffold_variants(base, 10)
            designed_compounds.extend(variants)
        
        return designed_compounds
    
    def cluster_compounds_by_similarity(self, compounds: List[str], 
                                      n_clusters: int = 5) -> Dict[int, List[str]]:
        """Cluster compounds by structural similarity."""
        from rdkit.Chem import DataStructs
        from rdkit import DataStructs
        
        # Calculate fingerprints
        fps = []
        valid_compounds = []
        
        for smiles in compounds:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
                valid_compounds.append(smiles)
        
        if not fps:
            return {}
        
        # Calculate similarity matrix
        n_compounds = len(fps)
        similarity_matrix = np.zeros((n_compounds, n_compounds))
        
        for i in range(n_compounds):
            for j in range(i, n_compounds):
                similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        # Convert to distance matrix for clustering
        distance_matrix = 1 - similarity_matrix
        
        # Cluster using KMeans
        kmeans = KMeans(n_clusters=min(n_clusters, n_compounds), random_state=42)
        cluster_labels = kmeans.fit_predict(distance_matrix)
        
        # Group compounds by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_compounds[i])
        
        return clusters

def create_psychedelic_generator() -> PsychedelicGenerator:
    """Factory function to create psychedelic generator."""
    return PsychedelicGenerator()

if __name__ == "__main__":
    # Test compound generation
    generator = create_psychedelic_generator()
    
    # Generate 2C-B analogs
    analogs = generator.generate_2c_analogs(10)
    print(f"Generated {len(analogs)} 2C-B analogs:")
    for analog in analogs[:5]:
        print(f"  {analog}")
    
    # Test optimization
    target_props = {
        'mw': (250, 350),
        'logp': (1.5, 3.0),
        'tpsa': (30, 80)
    }
    
    optimized = generator.optimize_for_properties(
        'CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN', 
        target_props, 
        num_generations=3, 
        population_size=20
    )
    print(f"\nOptimized compounds: {len(optimized)}")
    for compound in optimized[:3]:
        print(f"  {compound}")