"""
Structural Analysis Module
=========================

Tools for 5-HT2A receptor binding pocket analysis, allosteric site identification,
and structure-based drug design for psychedelic therapeutics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import os
import subprocess
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
import deepchem as dc
from Bio.PDB import PDBParser, PDBIO, Select
import requests
import tempfile

class HTR2AStructureAnalyzer:
    """5-HT2A receptor structure analyzer."""
    
    def __init__(self, data_dir: str = "./structural_data"):
        """
        Initialize structure analyzer.
        
        Args:
            data_dir: Directory to store structural data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.receptor_pdb = None
        self.binding_sites = []
        self.allosteric_sites = []
        
    def download_5ht2a_structure(self, pdb_id: str = "6A93") -> str:
        """
        Download 5-HT2A receptor structure from PDB.
        
        Args:
            pdb_id: PDB ID for 5-HT2A structure (6A93 is 5-HT2A with risperidone)
        """
        pdb_file = self.data_dir / f"{pdb_id.lower()}.pdb"
        
        if pdb_file.exists():
            print(f"PDB file {pdb_id} already exists")
            self.receptor_pdb = str(pdb_file)
            return str(pdb_file)
        
        # Download from RCSB PDB
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(pdb_file, 'w') as f:
                f.write(response.text)
            
            print(f"Downloaded {pdb_id} structure to {pdb_file}")
            self.receptor_pdb = str(pdb_file)
            return str(pdb_file)
            
        except requests.RequestException as e:
            print(f"Error downloading PDB {pdb_id}: {e}")
            return None
    
    def analyze_binding_pocket(self, ligand_resname: str = "9EM") -> Dict[str, any]:
        """
        Analyze the binding pocket around a ligand.
        
        Args:
            ligand_resname: Residue name of the ligand in PDB
        """
        if not self.receptor_pdb:
            raise ValueError("No receptor structure loaded")
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("receptor", self.receptor_pdb)
        
        # Find ligand
        ligand_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() == ligand_resname:
                        ligand_atoms.extend(residue.get_atoms())
        
        if not ligand_atoms:
            print(f"Warning: Ligand {ligand_resname} not found in structure")
            return {}
        
        # Find nearby residues (within 5Å of ligand)
        binding_residues = []
        cutoff = 5.0
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() != ligand_resname:
                        for atom in residue.get_atoms():
                            for ligand_atom in ligand_atoms:
                                distance = atom - ligand_atom
                                if distance < cutoff:
                                    binding_residues.append({
                                        'chain': chain.id,
                                        'residue': residue.get_resname(),
                                        'number': residue.id[1],
                                        'distance': distance
                                    })
                                    break
        
        # Remove duplicates
        unique_residues = []
        seen = set()
        for res in binding_residues:
            key = (res['chain'], res['residue'], res['number'])
            if key not in seen:
                unique_residues.append(res)
                seen.add(key)
        
        # Calculate binding pocket properties
        pocket_analysis = {
            'ligand_resname': ligand_resname,
            'binding_residues': unique_residues,
            'pocket_size': len(unique_residues),
            'hydrophobic_residues': [r for r in unique_residues 
                                   if r['residue'] in ['ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'TYR']],
            'polar_residues': [r for r in unique_residues 
                             if r['residue'] in ['SER', 'THR', 'ASN', 'GLN', 'HIS', 'TRP', 'TYR']],
            'charged_residues': [r for r in unique_residues 
                               if r['residue'] in ['ASP', 'GLU', 'LYS', 'ARG']],
        }
        
        self.binding_sites.append(pocket_analysis)
        return pocket_analysis
    
    def run_fpocket_analysis(self) -> Optional[Dict]:
        """
        Run fpocket for cavity detection and druggability assessment.
        Note: Requires fpocket to be installed and in PATH.
        """
        if not self.receptor_pdb:
            raise ValueError("No receptor structure loaded")
        
        fpocket_dir = self.data_dir / "fpocket_output"
        fpocket_dir.mkdir(exist_ok=True)
        
        try:
            # Run fpocket
            cmd = f"fpocket -f {self.receptor_pdb} -d {fpocket_dir}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Fpocket analysis completed successfully")
                return self._parse_fpocket_output(fpocket_dir)
            else:
                print(f"Fpocket failed: {result.stderr}")
                return None
                
        except FileNotFoundError:
            print("Fpocket not found. Please install fpocket and add to PATH.")
            return None
    
    def _parse_fpocket_output(self, fpocket_dir: Path) -> Dict:
        """Parse fpocket output files."""
        info_file = fpocket_dir / f"{Path(self.receptor_pdb).stem}_info.txt"
        
        if not info_file.exists():
            return {}
        
        pockets = []
        current_pocket = {}
        
        try:
            with open(info_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Pocket"):
                        if current_pocket:
                            pockets.append(current_pocket)
                        current_pocket = {'id': line}
                    elif ":" in line and current_pocket:
                        key, value = line.split(":", 1)
                        current_pocket[key.strip()] = value.strip()
                
                if current_pocket:
                    pockets.append(current_pocket)
        
        except Exception as e:
            print(f"Error parsing fpocket output: {e}")
            return {}
        
        return {
            'pockets': pockets,
            'num_pockets': len(pockets),
            'analysis_dir': str(fpocket_dir)
        }
    
    def identify_allosteric_sites(self, main_pocket_id: int = 0) -> List[Dict]:
        """
        Identify potential allosteric sites.
        
        Args:
            main_pocket_id: ID of the main binding pocket
        """
        fpocket_results = self.run_fpocket_analysis()
        
        if not fpocket_results or 'pockets' not in fpocket_results:
            return []
        
        allosteric_candidates = []
        
        for i, pocket in enumerate(fpocket_results['pockets']):
            if i != main_pocket_id:  # Skip main binding site
                # Evaluate druggability
                try:
                    volume = float(pocket.get('Volume', '0'))
                    druggability = float(pocket.get('Druggability Score', '0'))
                    
                    if volume > 200 and druggability > 0.5:  # Arbitrary thresholds
                        allosteric_candidates.append({
                            'pocket_id': i,
                            'volume': volume,
                            'druggability': druggability,
                            'pocket_info': pocket
                        })
                except ValueError:
                    continue
        
        # Sort by druggability score
        allosteric_candidates.sort(key=lambda x: x['druggability'], reverse=True)
        
        self.allosteric_sites = allosteric_candidates
        return allosteric_candidates
    
    def dock_ligand(self, ligand_smiles: str, pocket_center: Tuple[float, float, float] = None) -> Dict:
        """
        Perform molecular docking of a ligand.
        Note: This is a simplified interface - real docking would use AutoDock Vina or similar.
        """
        if not self.receptor_pdb:
            raise ValueError("No receptor structure loaded")
        
        # Generate 3D structure for ligand
        mol = Chem.MolFromSmiles(ligand_smiles)
        if mol is None:
            return {'error': 'Invalid SMILES'}
        
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Save ligand as SDF
        ligand_file = self.data_dir / "ligand.sdf"
        writer = Chem.SDWriter(str(ligand_file))
        writer.write(mol)
        writer.close()
        
        # This would normally call AutoDock Vina or similar
        # For now, return a placeholder
        docking_result = {
            'ligand_smiles': ligand_smiles,
            'ligand_file': str(ligand_file),
            'receptor_file': self.receptor_pdb,
            'status': 'prepared',
            'note': 'Docking requires external software like AutoDock Vina'
        }
        
        return docking_result
    
    def calculate_binding_affinity_descriptors(self, binding_pocket: Dict) -> Dict[str, float]:
        """Calculate descriptors for binding affinity prediction."""
        descriptors = {
            'pocket_size': binding_pocket.get('pocket_size', 0),
            'hydrophobic_fraction': len(binding_pocket.get('hydrophobic_residues', [])) / max(1, binding_pocket.get('pocket_size', 1)),
            'polar_fraction': len(binding_pocket.get('polar_residues', [])) / max(1, binding_pocket.get('pocket_size', 1)),
            'charged_fraction': len(binding_pocket.get('charged_residues', [])) / max(1, binding_pocket.get('pocket_size', 1)),
        }
        
        # Key residues for 5-HT2A binding
        key_residues_5ht2a = ['ASP155', 'SER159', 'PHE340', 'ASN343', 'SER239']
        
        descriptors['key_residue_coverage'] = sum(
            1 for res in binding_pocket.get('binding_residues', [])
            if f"{res['residue']}{res['number']}" in key_residues_5ht2a
        ) / len(key_residues_5ht2a)
        
        return descriptors

class PsychedelicDockingAnalyzer:
    """Specialized docking analyzer for psychedelic compounds."""
    
    def __init__(self, structure_analyzer: HTR2AStructureAnalyzer):
        self.structure_analyzer = structure_analyzer
        
    def dock_2cb_analogs(self, analog_smiles: List[str]) -> pd.DataFrame:
        """Dock 2C-B analogs and analyze binding modes."""
        results = []
        
        for smiles in analog_smiles:
            docking_result = self.structure_analyzer.dock_ligand(smiles)
            
            # Calculate molecular descriptors
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                descriptors = {
                    'smiles': smiles,
                    'mw': Chem.Descriptors.MolWt(mol),
                    'logp': Chem.Descriptors.MolLogP(mol),
                    'hbd': Chem.Descriptors.NumHDonors(mol),
                    'hba': Chem.Descriptors.NumHAcceptors(mol),
                    'docking_status': docking_result.get('status', 'failed')
                }
                results.append(descriptors)
        
        return pd.DataFrame(results)
    
    def analyze_binding_modes(self, compound_name: str, binding_pocket: Dict) -> Dict:
        """Analyze binding modes for psychedelic compounds."""
        analysis = {
            'compound': compound_name,
            'pocket_analysis': binding_pocket,
            'predicted_interactions': self._predict_interactions(binding_pocket),
            'druggability_assessment': self._assess_druggability(binding_pocket)
        }
        
        return analysis
    
    def _predict_interactions(self, binding_pocket: Dict) -> List[str]:
        """Predict key molecular interactions."""
        interactions = []
        
        # Check for key 5-HT2A interactions
        binding_residues = binding_pocket.get('binding_residues', [])
        
        for residue in binding_residues:
            res_name = residue['residue']
            res_num = residue['number']
            
            if res_name == 'ASP' and res_num == 155:
                interactions.append("Salt bridge with Asp155 (critical for agonist binding)")
            elif res_name == 'SER' and res_num in [159, 239]:
                interactions.append(f"Hydrogen bond with Ser{res_num}")
            elif res_name == 'PHE' and res_num == 340:
                interactions.append("π-π stacking with Phe340")
            elif res_name in ['TYR', 'PHE', 'TRP']:
                interactions.append(f"Aromatic interaction with {res_name}{res_num}")
        
        return interactions
    
    def _assess_druggability(self, binding_pocket: Dict) -> Dict[str, float]:
        """Assess druggability of the binding pocket."""
        pocket_size = binding_pocket.get('pocket_size', 0)
        hydrophobic_fraction = len(binding_pocket.get('hydrophobic_residues', [])) / max(1, pocket_size)
        
        assessment = {
            'size_score': min(1.0, pocket_size / 20.0),  # Normalize to 20 residues
            'hydrophobicity_score': hydrophobic_fraction,
            'druggability_score': (min(1.0, pocket_size / 20.0) + hydrophobic_fraction) / 2
        }
        
        return assessment

def create_structure_analyzer(data_dir: str = "./structural_data") -> HTR2AStructureAnalyzer:
    """Factory function to create structure analyzer."""
    return HTR2AStructureAnalyzer(data_dir)

if __name__ == "__main__":
    # Test structure analyzer
    analyzer = create_structure_analyzer()
    
    # Try to download 5-HT2A structure
    pdb_file = analyzer.download_5ht2a_structure("6A93")
    
    if pdb_file:
        print(f"Successfully downloaded structure: {pdb_file}")
        
        # Analyze binding pocket
        pocket_analysis = analyzer.analyze_binding_pocket("9EM")  # Risperidone
        print(f"Found binding pocket with {pocket_analysis.get('pocket_size', 0)} residues")
    else:
        print("Failed to download structure")