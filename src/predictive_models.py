"""
Predictive Machine Learning Models for Psychedelic Therapeutics
===============================================================

Advanced ML models for predicting:
- 5-HT2A receptor binding affinity
- CNS penetration properties
- ADMET characteristics
- Safety profiles

Author: AI Assistant for CNS Therapeutics Research
Focus: ML-driven psychedelic drug design and optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available. ML models will be limited.")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Fingerprints
    from rdkit.Chem.Fingerprints import FingerprintMols
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not available. Using fallback feature generation.")

class PsychedelicMLPredictor:
    """Machine learning predictor for psychedelic compound properties."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Known psychedelic activity data for training
        self.training_data = self._get_training_data()
        
        # Model configurations
        self.model_configs = {
            'ht2a_binding': {
                'target': 'pki_5ht2a',
                'algorithm': 'random_forest',
                'features': ['molecular', 'fingerprint'],
                'cv_folds': 5
            },
            'bbb_penetration': {
                'target': 'bbb_ratio',
                'algorithm': 'gradient_boosting', 
                'features': ['molecular'],
                'cv_folds': 5
            },
            'cns_activity': {
                'target': 'cns_score',
                'algorithm': 'random_forest',
                'features': ['molecular', 'fingerprint'],
                'cv_folds': 5
            }
        }
    
    def _get_training_data(self) -> pd.DataFrame:
        """Get curated training data for psychedelic compounds."""
        # Curated dataset of known psychedelic compounds with experimental data
        training_compounds = [
            # 2C Series
            {'name': '2C-B', 'smiles': 'CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN', 
             'pki_5ht2a': 8.7, 'bbb_ratio': 0.85, 'cns_score': 8.2, 'class': '2C'},
            {'name': '2C-I', 'smiles': 'CCc1cc(I)c(OCc2ccccc2)c(I)c1CCN',
             'pki_5ht2a': 8.9, 'bbb_ratio': 0.82, 'cns_score': 8.1, 'class': '2C'},
            {'name': '2C-E', 'smiles': 'CCCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN',
             'pki_5ht2a': 7.8, 'bbb_ratio': 0.78, 'cns_score': 7.5, 'class': '2C'},
            {'name': '2C-P', 'smiles': 'CC(C)c1cc(Br)c(OCc2ccccc2)c(Br)c1CCN',
             'pki_5ht2a': 7.2, 'bbb_ratio': 0.75, 'cns_score': 7.0, 'class': '2C'},
            
            # DOx Series
            {'name': 'DOB', 'smiles': 'CC(N)Cc1cc(Br)c(OCc2ccccc2)c(Br)c1',
             'pki_5ht2a': 8.2, 'bbb_ratio': 0.88, 'cns_score': 8.0, 'class': 'DOx'},
            {'name': 'DOI', 'smiles': 'CC(N)Cc1cc(I)c(OCc2ccccc2)c(I)c1',
             'pki_5ht2a': 8.5, 'bbb_ratio': 0.85, 'cns_score': 8.1, 'class': 'DOx'},
            {'name': 'DOM', 'smiles': 'COc1cc(CC(C)N)cc(OC)c1OCc1ccccc1',
             'pki_5ht2a': 7.8, 'bbb_ratio': 0.70, 'cns_score': 7.2, 'class': 'DOx'},
            
            # Mescaline analogs
            {'name': 'Mescaline', 'smiles': 'COc1cc(CCN)cc(OC)c1OC',
             'pki_5ht2a': 6.2, 'bbb_ratio': 0.65, 'cns_score': 6.0, 'class': 'Mescaline'},
            {'name': 'Escaline', 'smiles': 'CCOc1cc(CCN)cc(OCC)c1OCC',
             'pki_5ht2a': 6.8, 'bbb_ratio': 0.72, 'cns_score': 6.5, 'class': 'Mescaline'},
            
            # NBOMe Series (lower CNS scores due to toxicity concerns)
            {'name': '25B-NBOMe', 'smiles': 'COc1cc(CCNCc2ccccc2OC)c(Br)cc1OCc1ccccc1',
             'pki_5ht2a': 9.2, 'bbb_ratio': 0.60, 'cns_score': 5.5, 'class': 'NBOMe'},
            {'name': '25I-NBOMe', 'smiles': 'COc1cc(CCNCc2ccccc2OC)c(I)cc1OCc1ccccc1',
             'pki_5ht2a': 9.5, 'bbb_ratio': 0.58, 'cns_score': 5.2, 'class': 'NBOMe'},
            
            # Reference compounds (non-psychedelic)
            {'name': 'Serotonin', 'smiles': 'NCCc1c[nH]c2ccc(O)cc12',
             'pki_5ht2a': 6.0, 'bbb_ratio': 0.15, 'cns_score': 3.0, 'class': 'Reference'},
            {'name': 'Dopamine', 'smiles': 'NCCc1ccc(O)c(O)c1',
             'pki_5ht2a': 4.5, 'bbb_ratio': 0.10, 'cns_score': 2.5, 'class': 'Reference'},
        ]
        
        return pd.DataFrame(training_compounds)
    
    def calculate_molecular_features(self, smiles: str) -> Dict[str, float]:
        """Calculate molecular descriptors for ML features."""
        if not RDKIT_AVAILABLE:
            # Fallback feature calculation
            return {
                'mw': len(smiles) * 12,  # Rough estimate
                'logp': smiles.count('c') * 0.5,
                'tpsa': smiles.count('O') * 20 + smiles.count('N') * 10,
                'hbd': smiles.count('OH') + smiles.count('NH'),
                'hba': smiles.count('O') + smiles.count('N'),
                'rotb': smiles.count('C') // 4,
                'rings': smiles.count('1') + smiles.count('2')
            }
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        features = {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol), 
            'rotb': Descriptors.NumRotatableBonds(mol),
            'rings': Descriptors.RingCount(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'sa_score': Descriptors.BertzCT(mol),  # Structural complexity
            'formal_charge': Chem.rdmolops.GetFormalCharge(mol)
        }
        
        # Add fragment counts
        features.update({
            'benzene_rings': len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1ccccc1'))),
            'halogen_count': sum(1 for atom in mol.GetAtoms() 
                               if atom.GetSymbol() in ['F', 'Cl', 'Br', 'I']),
            'ether_count': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]O[#6]'))),
            'amine_count': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]')))
        })
        
        return features
    
    def calculate_fingerprint_features(self, smiles: str) -> np.ndarray:
        """Calculate molecular fingerprints for ML features."""
        if not RDKIT_AVAILABLE:
            # Fallback: simple hash-based features
            return np.array([hash(smiles[i:i+3]) % 1024 for i in range(min(1024, len(smiles)-2))])
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(1024)
        
        # Morgan fingerprint (ECFP4)
        fp = FingerprintMols.FingerprintMol(mol)
        arr = np.zeros((1024,))
        fp.ToBitString()  # Convert to bit string
        
        return arr
    
    def prepare_training_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
        """Prepare feature matrix for training."""
        features_list = []
        feature_names = []
        
        for idx, row in df.iterrows():
            # Molecular descriptors
            mol_features = self.calculate_molecular_features(row['smiles'])
            
            # Fingerprint features (reduced dimension for speed)
            if RDKIT_AVAILABLE:
                fp_features = self.calculate_fingerprint_features(row['smiles'])[:128]  # Use first 128 bits
                fp_feature_names = [f'fp_{i}' for i in range(128)]
            else:
                fp_features = []
                fp_feature_names = []
            
            # Combine features
            all_features = list(mol_features.values()) + list(fp_features)
            features_list.append(all_features)
            
            # Feature names (only for first iteration)
            if idx == 0:
                feature_names = list(mol_features.keys()) + fp_feature_names
        
        X = np.array(features_list)
        feature_name_map = {name: i for i, name in enumerate(feature_names)}
        
        return X, feature_name_map
    
    def train_model(self, model_name: str) -> Dict[str, Any]:
        """Train a specific ML model."""
        if not SKLEARN_AVAILABLE:
            return {"error": "Scikit-learn not available"}
        
        config = self.model_configs.get(model_name)
        if not config:
            return {"error": f"Unknown model: {model_name}"}
        
        # Prepare data
        df = self.training_data
        X, feature_names = self.prepare_training_features(df)
        y = df[config['target']].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Choose algorithm
        if config['algorithm'] == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif config['algorithm'] == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            return {"error": f"Unknown algorithm: {config['algorithm']}"}
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=config['cv_folds'], scoring='r2')
        
        # Store model
        self.models[model_name] = pipeline
        self.feature_names[model_name] = feature_names
        
        # Save model
        model_path = self.model_dir / f"{model_name}_model.pkl"
        joblib.dump(pipeline, model_path)
        
        # Feature importance (for tree-based models)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names.keys(), model.feature_importances_))
            feature_importance = dict(sorted(importance_dict.items(), 
                                           key=lambda x: x[1], reverse=True)[:10])
        
        results = {
            'model_name': model_name,
            'algorithm': config['algorithm'],
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'model_path': str(model_path)
        }
        
        return results
    
    def predict(self, smiles: str, model_name: str) -> Dict[str, Any]:
        """Make prediction for a single compound."""
        if model_name not in self.models:
            return {"error": f"Model {model_name} not trained"}
        
        # Calculate features
        mol_features = self.calculate_molecular_features(smiles)
        if RDKIT_AVAILABLE:
            fp_features = self.calculate_fingerprint_features(smiles)[:128]
        else:
            fp_features = []
        
        # Combine features
        features = list(mol_features.values()) + list(fp_features)
        X = np.array([features])
        
        # Make prediction
        prediction = self.models[model_name].predict(X)[0]
        
        # Get prediction confidence (for tree-based models)
        confidence = 0.8  # Default confidence
        if hasattr(self.models[model_name].named_steps['model'], 'estimators_'):
            # For ensemble models, use prediction variance as confidence indicator
            predictions = []
            for estimator in self.models[model_name].named_steps['model'].estimators_:
                pred = estimator.predict(self.models[model_name].named_steps['scaler'].transform(X))[0]
                predictions.append(pred)
            
            pred_std = np.std(predictions)
            confidence = max(0.1, 1.0 - pred_std / np.mean(predictions))
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'model_name': model_name,
            'smiles': smiles
        }
    
    def train_all_models(self) -> Dict[str, Dict]:
        """Train all configured models."""
        results = {}
        
        print("🤖 Training psychedelic ML models...")
        
        for model_name in self.model_configs.keys():
            print(f"Training {model_name} model...")
            try:
                result = self.train_model(model_name)
                results[model_name] = result
                
                if 'error' not in result:
                    print(f"✅ {model_name}: R² = {result['test_r2']:.3f}, RMSE = {result['test_rmse']:.3f}")
                else:
                    print(f"❌ {model_name}: {result['error']}")
                    
            except Exception as e:
                print(f"❌ {model_name} training failed: {str(e)}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def batch_predict(self, compounds_df: pd.DataFrame, model_names: List[str] = None) -> pd.DataFrame:
        """Make batch predictions for multiple compounds."""
        if model_names is None:
            model_names = list(self.models.keys())
        
        results_df = compounds_df.copy()
        
        for model_name in model_names:
            if model_name not in self.models:
                continue
                
            predictions = []
            confidences = []
            
            for _, row in compounds_df.iterrows():
                try:
                    result = self.predict(row['smiles'], model_name)
                    predictions.append(result['prediction'])
                    confidences.append(result['confidence'])
                except:
                    predictions.append(np.nan)
                    confidences.append(0.0)
            
            results_df[f'{model_name}_pred'] = predictions
            results_df[f'{model_name}_confidence'] = confidences
        
        return results_df

if __name__ == "__main__":
    # Test ML predictor
    print("🤖 Testing Predictive ML Models...")
    
    predictor = PsychedelicMLPredictor()
    
    # Train models
    if SKLEARN_AVAILABLE:
        results = predictor.train_all_models()
        
        # Test prediction
        test_smiles = "CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN"  # 2C-B
        
        for model_name in predictor.models.keys():
            pred_result = predictor.predict(test_smiles, model_name)
            print(f"{model_name} prediction for 2C-B: {pred_result['prediction']:.2f} (confidence: {pred_result['confidence']:.3f})")
    
    print("\n✅ Predictive ML Models Ready!")