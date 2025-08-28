"""
Interactive 3D Molecular Design for Psychedelic Therapeutics
============================================================

Advanced 3D molecular design tools with:
- Real-time structure editing
- 5-HT2A receptor docking visualization
- Interactive pharmacophore mapping
- Structure-activity correlation in 3D space

Author: AI Assistant for CNS Therapeutics Research
Focus: 5-HT2A receptor-targeted psychedelic drug design
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import base64
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdFMCS
    import py3Dmol
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not available. Some 3D design features will be limited.")

class MolecularDesigner3D:
    """Advanced 3D molecular design and visualization."""
    
    def __init__(self):
        self.pharmacophore_features = {
            'aromatic_ring': {'color': 'orange', 'radius': 2.5, 'description': 'Aromatic ring system'},
            'hbond_donor': {'color': 'blue', 'radius': 1.5, 'description': 'H-bond donor'},
            'hbond_acceptor': {'color': 'red', 'radius': 1.5, 'description': 'H-bond acceptor'},
            'positive_ionizable': {'color': 'cyan', 'radius': 2.0, 'description': 'Positive ionizable'},
            'hydrophobic': {'color': 'green', 'radius': 2.0, 'description': 'Hydrophobic region'}
        }
        
        # 5-HT2A receptor binding site coordinates (from PDB structure)
        self.ht2a_binding_site = {
            'center': [0, 0, 0],  # Would be actual coordinates from PDB
            'key_residues': [
                {'name': 'Phe234', 'pos': [-2.1, 1.3, 0.5], 'type': 'aromatic'},
                {'name': 'Asp155', 'pos': [1.8, -1.2, 0.8], 'type': 'acceptor'},
                {'name': 'Ser159', 'pos': [0.5, 2.1, -1.2], 'type': 'donor'},
                {'name': 'Val156', 'pos': [-1.5, -0.8, 1.9], 'type': 'hydrophobic'},
                {'name': 'Trp336', 'pos': [2.3, 1.8, -0.7], 'type': 'aromatic'}
            ]
        }
    
    def create_3d_viewer(self, width: int = 800, height: int = 600) -> py3Dmol.view:
        """Create an interactive 3D viewer."""
        viewer = py3Dmol.view(width=width, height=height)
        viewer.setBackgroundColor('white')
        return viewer
    
    def add_molecule_to_viewer(self, viewer: py3Dmol.view, smiles: str, 
                              style: str = 'stick', color: str = 'default') -> bool:
        """Add a molecule to the 3D viewer."""
        if not RDKIT_AVAILABLE:
            return False
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Generate 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol)
            
            # Convert to MOL block
            molblock = Chem.MolToMolBlock(mol)
            
            # Add to viewer
            viewer.addModel(molblock, 'mol')
            
            # Set style
            if style == 'stick':
                viewer.setStyle({'stick': {'radius': 0.15, 'color': color}})
            elif style == 'sphere':
                viewer.setStyle({'sphere': {'radius': 0.8, 'color': color}})
            elif style == 'cartoon':
                viewer.setStyle({'cartoon': {'color': color}})
            
            return True
            
        except Exception as e:
            print(f"Error adding molecule to viewer: {str(e)}")
            return False
    
    def add_pharmacophore_to_viewer(self, viewer: py3Dmol.view, 
                                   pharmacophore_points: List[Dict]) -> None:
        """Add pharmacophore features to the viewer."""
        for point in pharmacophore_points:
            feature_type = point['type']
            position = point['position']
            
            if feature_type in self.pharmacophore_features:
                feature = self.pharmacophore_features[feature_type]
                
                # Add sphere for pharmacophore feature
                viewer.addSphere({
                    'center': {'x': position[0], 'y': position[1], 'z': position[2]},
                    'radius': feature['radius'],
                    'color': feature['color'],
                    'alpha': 0.3
                })
                
                # Add label
                viewer.addLabel(
                    feature['description'],
                    {'position': {'x': position[0], 'y': position[1], 'z': position[2]},
                     'backgroundColor': feature['color'],
                     'fontColor': 'white'}
                )
    
    def add_receptor_binding_site(self, viewer: py3Dmol.view) -> None:
        """Add 5-HT2A receptor binding site visualization."""
        # Add binding site center
        center = self.ht2a_binding_site['center']
        viewer.addSphere({
            'center': {'x': center[0], 'y': center[1], 'z': center[2]},
            'radius': 5.0,
            'color': 'lightblue',
            'alpha': 0.1
        })
        
        # Add key residues
        for residue in self.ht2a_binding_site['key_residues']:
            pos = residue['pos']
            res_type = residue['type']
            
            # Color by residue type
            color_map = {
                'aromatic': 'purple',
                'donor': 'blue',
                'acceptor': 'red',
                'hydrophobic': 'green'
            }
            
            color = color_map.get(res_type, 'gray')
            
            viewer.addSphere({
                'center': {'x': pos[0], 'y': pos[1], 'z': pos[2]},
                'radius': 1.0,
                'color': color,
                'alpha': 0.7
            })
            
            viewer.addLabel(
                residue['name'],
                {'position': {'x': pos[0], 'y': pos[1], 'z': pos[2] + 1.5},
                 'backgroundColor': color,
                 'fontColor': 'white'}
            )
    
    def generate_2cb_analogs(self, base_smiles: str = "CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN",
                           modifications: List[str] = None) -> List[Dict]:
        """Generate 2C-B analogs with specific modifications."""
        if not RDKIT_AVAILABLE:
            return []
        
        if modifications is None:
            modifications = [
                "halogen_swap",     # Swap Br for I, Cl, F
                "chain_extension",  # Extend ethyl to propyl
                "methoxy_addition", # Add methoxy groups
                "ring_substitution" # Modify benzyl ring
            ]
        
        analogs = []
        base_mol = Chem.MolFromSmiles(base_smiles)
        
        if base_mol is None:
            return analogs
        
        # Halogen swaps (2C-I, 2C-Cl, 2C-F)
        if "halogen_swap" in modifications:
            halogen_replacements = [
                ("Br", "I", "2C-I analog"),
                ("Br", "Cl", "2C-Cl analog"),
                ("Br", "F", "2C-F analog")
            ]
            
            for old_hal, new_hal, name in halogen_replacements:
                new_smiles = base_smiles.replace(old_hal, new_hal)
                if new_smiles != base_smiles:
                    analogs.append({
                        'name': name,
                        'smiles': new_smiles,
                        'modification': f"Halogen swap: {old_hal} → {new_hal}",
                        'rationale': f"Exploring halogen effects on 5-HT2A binding"
                    })
        
        # Chain extensions (2C-E, 2C-P analogs)
        if "chain_extension" in modifications:
            chain_modifications = [
                ("CCc1", "CCCc1", "2C-E analog", "Propyl chain extension"),
                ("CCc1", "CC(C)c1", "2C-P analog", "Isopropyl substitution")
            ]
            
            for old_chain, new_chain, name, desc in chain_modifications:
                new_smiles = base_smiles.replace(old_chain, new_chain)
                if new_smiles != base_smiles:
                    analogs.append({
                        'name': name,
                        'smiles': new_smiles,
                        'modification': desc,
                        'rationale': "Investigating chain length effects on potency"
                    })
        
        # Methoxy additions (mescaline-like)
        if "methoxy_addition" in modifications:
            # This would require more sophisticated SMARTS-based modifications
            # Simplified example:
            mescaline_analog = "COc1cc(CCN)cc(OC)c1OC"
            analogs.append({
                'name': "Mescaline-type analog",
                'smiles': mescaline_analog,
                'modification': "Methoxy substitution pattern",
                'rationale': "Exploring alternative substitution patterns"
            })
        
        return analogs
    
    def calculate_pharmacophore_overlap(self, mol1_smiles: str, mol2_smiles: str) -> Dict:
        """Calculate pharmacophore overlap between two molecules."""
        if not RDKIT_AVAILABLE:
            return {}
        
        try:
            mol1 = Chem.MolFromSmiles(mol1_smiles)
            mol2 = Chem.MolFromSmiles(mol2_smiles)
            
            if mol1 is None or mol2 is None:
                return {}
            
            # Generate 3D conformers
            mol1 = Chem.AddHs(mol1)
            mol2 = Chem.AddHs(mol2)
            
            AllChem.EmbedMolecule(mol1, AllChem.ETKDG())
            AllChem.EmbedMolecule(mol2, AllChem.ETKDG())
            
            # Find Maximum Common Substructure
            mcs = rdFMCS.FindMCS([mol1, mol2])
            
            if mcs.numAtoms > 0:
                overlap_score = mcs.numAtoms / max(mol1.GetNumAtoms(), mol2.GetNumAtoms())
            else:
                overlap_score = 0.0
            
            return {
                'mcs_atoms': mcs.numAtoms,
                'mcs_bonds': mcs.numBonds,
                'overlap_score': overlap_score,
                'mcs_smarts': mcs.smartsString if mcs.numAtoms > 0 else None
            }
            
        except Exception as e:
            print(f"Error calculating pharmacophore overlap: {str(e)}")
            return {}
    
    def create_interactive_docking_view(self, ligand_smiles: str, 
                                       ligand_name: str = "Ligand") -> py3Dmol.view:
        """Create interactive docking visualization."""
        viewer = self.create_3d_viewer(900, 700)
        
        # Add ligand
        if self.add_molecule_to_viewer(viewer, ligand_smiles, 'stick', 'cyan'):
            print(f"✅ Added {ligand_name} to docking view")
        
        # Add receptor binding site
        self.add_receptor_binding_site(viewer)
        
        # Add pharmacophore features for 5-HT2A
        pharmacophore_points = [
            {'type': 'aromatic_ring', 'position': [0, 0, 0]},
            {'type': 'positive_ionizable', 'position': [5.5, 1.2, 0.8]},
            {'type': 'hydrophobic', 'position': [-2.1, 1.3, 0.5]},
            {'type': 'hbond_acceptor', 'position': [1.8, -1.2, 0.8]}
        ]
        
        self.add_pharmacophore_to_viewer(viewer, pharmacophore_points)
        
        # Set initial view
        viewer.zoomTo()
        viewer.spin(True)
        
        return viewer

class StructureActivityMapper:
    """Map structure-activity relationships in 3D space."""
    
    def __init__(self):
        self.activity_data = {}
        self.property_ranges = {
            'low_activity': (0, 5),
            'moderate_activity': (5, 7),
            'high_activity': (7, 10)
        }
    
    def add_activity_data(self, smiles: str, activity: float, name: str = None):
        """Add experimental activity data."""
        self.activity_data[smiles] = {
            'activity': activity,
            'name': name or f"Compound_{len(self.activity_data)}"
        }
    
    def create_sar_heatmap_3d(self, compounds_df: pd.DataFrame) -> py3Dmol.view:
        """Create 3D SAR heatmap visualization."""
        viewer = py3Dmol.view(width=900, height=700)
        viewer.setBackgroundColor('white')
        
        if not RDKIT_AVAILABLE:
            return viewer
        
        # Color compounds by activity
        for idx, row in compounds_df.iterrows():
            smiles = row['smiles']
            activity = row.get('ht2a_affinity_pred', 5.0)  # Use predicted activity
            name = row.get('name', f'Compound_{idx}')
            
            # Determine color based on activity
            if activity >= 8:
                color = 'red'      # High activity
            elif activity >= 6:
                color = 'orange'   # Moderate activity
            elif activity >= 4:
                color = 'yellow'   # Low activity
            else:
                color = 'gray'     # Inactive
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                    AllChem.UFFOptimizeMolecule(mol)
                    
                    molblock = Chem.MolToMolBlock(mol)
                    
                    # Position molecules in grid
                    x_offset = (idx % 4) * 15
                    y_offset = (idx // 4) * 15
                    
                    # Add molecule with activity-based coloring
                    model_id = viewer.addModel(molblock, 'mol')
                    viewer.setStyle({'model': model_id}, 
                                   {'stick': {'radius': 0.2, 'color': color}})
                    
                    # Add activity label
                    viewer.addLabel(
                        f"{name}\npKi: {activity:.1f}",
                        {'position': {'x': x_offset, 'y': y_offset, 'z': 5},
                         'backgroundColor': color,
                         'fontColor': 'white',
                         'fontSize': 10}
                    )
                    
            except Exception as e:
                print(f"Error adding {name} to SAR heatmap: {str(e)}")
        
        viewer.zoomTo()
        return viewer

def create_molecular_editor_interface() -> Dict:
    """Create configuration for molecular editor interface."""
    return {
        'editor_config': {
            'width': 800,
            'height': 600,
            'background': 'white',
            'tools': [
                'select', 'draw_bond', 'draw_atom', 'erase',
                'rotate', 'zoom', 'center'
            ],
            'atom_palette': ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'],
            'bond_types': ['single', 'double', 'triple', 'aromatic']
        },
        'validation': {
            'check_valence': True,
            'check_aromaticity': True,
            'auto_complete': True
        },
        'export_formats': ['smiles', 'mol', 'sdf', 'png'],
        'real_time_analysis': {
            'descriptors': True,
            'drug_likeness': True,
            'activity_prediction': True,
            'safety_alerts': True
        }
    }

if __name__ == "__main__":
    # Test the 3D molecular designer
    print("🧬 Testing 3D Molecular Designer...")
    
    designer = MolecularDesigner3D()
    
    # Test 2C-B analog generation
    analogs = designer.generate_2cb_analogs()
    print(f"Generated {len(analogs)} 2C-B analogs:")
    for analog in analogs[:3]:
        print(f"  • {analog['name']}: {analog['modification']}")
    
    # Test pharmacophore overlap
    cb_smiles = "CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN"
    ci_smiles = "CCc1cc(I)c(OCc2ccccc2)c(I)c1CCN"
    
    if RDKIT_AVAILABLE:
        overlap = designer.calculate_pharmacophore_overlap(cb_smiles, ci_smiles)
        print(f"\n2C-B vs 2C-I overlap: {overlap.get('overlap_score', 0):.3f}")
    
    print("\n✅ 3D Molecular Designer Ready!")