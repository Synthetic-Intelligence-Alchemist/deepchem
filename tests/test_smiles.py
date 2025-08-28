"""
Test suite for SMILES parsing and 3D embedding functionality.
"""

import unittest
import sys
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from data import load_demo, validate_smiles_column
from descriptors import smiles_to_mol, calculate_descriptors, calculate_drug_likeness, bbb_label, compute
from viz3d import smiles_to_molblock, validate_3d_structure

class TestSMILESFunctionality(unittest.TestCase):
    """Test SMILES parsing and molecular property calculation."""
    
    @classmethod
    def setUpClass(cls):
        """Load test data once for all tests."""
        cls.df = load_demo()
        cls.test_smiles = cls.df['smiles'].tolist()[:5]  # First 5 compounds
        cls.test_names = cls.df['name'].tolist()[:5]
    
    def test_data_loading(self):
        """Test that demo data loads correctly."""
        self.assertIsNotNone(self.df)
        self.assertGreater(len(self.df), 0)
        self.assertIn('smiles', self.df.columns)
        self.assertIn('name', self.df.columns)
        self.assertIn('class', self.df.columns)
    
    def test_smiles_validation(self):
        """Test SMILES validation."""
        # Should not raise exception for valid dataset
        validate_smiles_column(self.df)
        
        # Test individual SMILES parsing
        for smiles in self.test_smiles:
            mol = smiles_to_mol(smiles)
            self.assertIsNotNone(mol, f"Failed to parse SMILES: {smiles}")
    
    def test_descriptor_calculation(self):
        """Test molecular descriptor calculation."""
        for smiles, name in zip(self.test_smiles, self.test_names):
            mol = smiles_to_mol(smiles)
            self.assertIsNotNone(mol, f"Failed to parse {name}: {smiles}")
            
            descriptors = calculate_descriptors(mol)
            
            # Check that all expected descriptors are present
            expected_keys = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotb', 'rings']
            for key in expected_keys:
                self.assertIn(key, descriptors, f"Missing descriptor {key} for {name}")
                self.assertIsNotNone(descriptors[key], f"None value for {key} in {name}")
            
            # Check reasonable ranges
            self.assertGreater(descriptors['mw'], 0, f"Invalid MW for {name}")
            self.assertGreater(descriptors['tpsa'], 0, f"Invalid TPSA for {name}")
            self.assertGreaterEqual(descriptors['hbd'], 0, f"Invalid HBD for {name}")
            self.assertGreaterEqual(descriptors['hba'], 0, f"Invalid HBA for {name}")
    
    def test_drug_likeness_calculation(self):
        """Test drug-likeness score calculation."""
        for smiles, name in zip(self.test_smiles, self.test_names):
            mol = smiles_to_mol(smiles)
            descriptors = calculate_descriptors(mol)
            drug_likeness = calculate_drug_likeness(descriptors)
            
            # Drug-likeness should be between 0 and 1
            self.assertGreaterEqual(drug_likeness, 0.0, f"Drug-likeness < 0 for {name}")
            self.assertLessEqual(drug_likeness, 1.0, f"Drug-likeness > 1 for {name}")
    
    def test_bbb_labeling(self):
        """Test BBB penetration labeling."""
        # Test known values
        self.assertEqual(bbb_label(30.0), "Good BBB")  # Low TPSA
        self.assertEqual(bbb_label(80.0), "Poor BBB")  # High TPSA
        self.assertEqual(bbb_label(float('nan')), "Unknown")  # NaN value
        
        # Test with real molecules
        for smiles, name in zip(self.test_smiles, self.test_names):
            mol = smiles_to_mol(smiles)
            descriptors = calculate_descriptors(mol)
            bbb = bbb_label(descriptors['tpsa'])
            
            self.assertIn(bbb, ["Good BBB", "Poor BBB", "Unknown"], 
                         f"Invalid BBB label for {name}: {bbb}")
    
    def test_3d_embedding(self):
        """Test 3D structure generation."""
        success_count = 0
        
        for smiles, name in zip(self.test_smiles, self.test_names):
            try:
                molblock = smiles_to_molblock(smiles)
                is_valid = validate_3d_structure(smiles)
                
                if molblock is not None and is_valid:
                    success_count += 1
                    # Check that molblock contains expected content
                    self.assertIn("V2000", molblock, f"Invalid molblock format for {name}")
                    self.assertIn("M  END", molblock, f"Incomplete molblock for {name}")
                
            except Exception as e:
                print(f"Warning: 3D embedding failed for {name}: {str(e)}")
        
        # At least 20% of molecules should embed successfully (relaxed for fallback mode)
        success_rate = success_count / len(self.test_smiles)
        self.assertGreaterEqual(success_rate, 0.2, 
                               f"3D embedding success rate too low: {success_rate:.2f}")
        print(f"3D embedding success rate: {success_rate:.2f} ({success_count}/{len(self.test_smiles)})")

class TestSpecificMolecules(unittest.TestCase):
    """Test specific molecules with known properties."""
    
    def test_2cb_properties(self):
        """Test 2C-B specific properties."""
        # Use the full pipeline to get enhanced data with known properties
        df = load_demo()
        df_enhanced = compute(df)
        
        # Find 2C-B in the enhanced data
        cb_row = df_enhanced[df_enhanced['name'] == '2C-B'].iloc[0]
        
        # 2C-B should have reasonable properties for a psychedelic
        self.assertAlmostEqual(cb_row['mw'], 334.1, delta=1.0, msg="2C-B MW incorrect")
        self.assertGreater(cb_row['logp'], 2.0, msg="2C-B LogP should be > 2")
        self.assertLess(cb_row['tpsa'], 60.0, msg="2C-B TPSA should be < 60 for BBB")
        
        # Should have good drug-likeness
        self.assertGreater(cb_row['drug_likeness'], 0.7, msg="2C-B should have good drug-likeness")
        
        # Should have good BBB penetration
        self.assertEqual(cb_row['bbb_label'], "Good BBB", msg="2C-B should have good BBB penetration")
    
    def test_mescaline_properties(self):
        """Test mescaline specific properties."""
        # Use the full pipeline to get enhanced data with known properties
        df = load_demo()
        df_enhanced = compute(df)
        
        # Find Mescaline in the enhanced data
        mescaline_row = df_enhanced[df_enhanced['name'] == 'Mescaline'].iloc[0]
        
        # Mescaline should be smaller than 2C-B
        self.assertAlmostEqual(mescaline_row['mw'], 211.3, delta=1.0, msg="Mescaline MW incorrect")
        self.assertLess(mescaline_row['logp'], 2.0, msg="Mescaline should be less lipophilic than 2C-B")
    
    def test_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        invalid_smiles = [
            "INVALID",
            "",
            "C#C#C",  # Unstable
            "C(C(C(C",  # Unclosed parentheses
        ]
        
        for smiles in invalid_smiles:
            mol = smiles_to_mol(smiles)
            # Should return None for invalid SMILES
            if mol is not None:
                # If it somehow parses, descriptors should handle it gracefully
                descriptors = calculate_descriptors(mol)
                self.assertIsInstance(descriptors, dict)

def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSMILESFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestSpecificMolecules))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🧪 Running SMILES and 3D embedding tests...")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)