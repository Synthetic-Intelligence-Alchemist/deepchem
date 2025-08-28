"""
SAR Prediction Models Module
============================

Graph neural networks and machine learning models for predicting 
5-HT2A receptor binding affinity and psychedelic activity.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import deepchem as dc
from deepchem.models import GraphConvModel, AttentiveFPModel, MultitaskRegressor
from deepchem.trans import NormalizationTransformer, BalancingTransformer
from deepchem.metrics import Metric, pearson_r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

class PsychedelicSARModel:
    """SAR prediction model for psychedelic compounds."""
    
    def __init__(self, model_type: str = 'attentivefp'):
        """
        Initialize SAR model.
        
        Args:
            model_type: Type of model ('graphconv', 'attentivefp', 'rf', 'multitask')
        """
        self.model_type = model_type
        self.model = None
        self.transformers = []
        self.trained = False
        
    def create_model(self, tasks: List[str], n_tasks: int = 1, mode: str = 'regression'):
        """Create the prediction model."""
        if self.model_type == 'graphconv':
            self.model = GraphConvModel(
                n_tasks=n_tasks,
                mode=mode,
                batch_size=32,
                learning_rate=0.001,
                graph_conv_layers=[64, 64],
                dense_layer_size=128,
                dropout=0.25
            )
        
        elif self.model_type == 'attentivefp':
            self.model = AttentiveFPModel(
                n_tasks=n_tasks,
                mode=mode,
                batch_size=32,
                learning_rate=0.001,
                num_layers=2,
                num_timesteps=2,
                graph_feat_size=200,
                dropout=0.1
            )
        
        elif self.model_type == 'multitask':
            self.model = MultitaskRegressor(
                n_tasks=n_tasks,
                n_features=2048,  # Assuming ECFP features
                layer_sizes=[1000, 500, 100],
                weight_init_stddevs=[0.02, 0.02, 0.02],
                bias_init_consts=[1.0, 1.0, 1.0],
                dropouts=[0.25, 0.25, 0.25],
                learning_rate=0.001
            )
        
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_data(self, dataset: dc.data.Dataset, 
                    validation_split: float = 0.2) -> Tuple[dc.data.Dataset, dc.data.Dataset]:
        """Prepare training and validation datasets."""
        # Create random split
        splitter = dc.splits.RandomSplitter()
        train_dataset, valid_dataset = splitter.train_valid_split(
            dataset, frac_valid=validation_split
        )
        
        # Apply transformers
        if self.model_type != 'rf':
            # Normalization for neural networks
            transformer = NormalizationTransformer(transform_y=True, dataset=train_dataset)
            train_dataset = transformer.transform(train_dataset)
            valid_dataset = transformer.transform(valid_dataset)
            self.transformers.append(transformer)
        
        return train_dataset, valid_dataset
    
    def train(self, train_dataset: dc.data.Dataset, 
             valid_dataset: dc.data.Dataset,
             nb_epoch: int = 100) -> Dict[str, float]:
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        if self.model_type == 'rf':
            # Train sklearn model
            X_train = train_dataset.X
            y_train = train_dataset.y.ravel()
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            
            X_valid = valid_dataset.X
            y_valid = valid_dataset.y.ravel()
            valid_score = self.model.score(X_valid, y_valid)
            
            metrics = {
                'train_r2': train_score,
                'valid_r2': valid_score
            }
            
        else:
            # Train DeepChem model
            self.model.fit(train_dataset, nb_epoch=nb_epoch)
            
            # Evaluate
            train_scores = self.model.evaluate(train_dataset, [dc.metrics.pearson_r2_score])
            valid_scores = self.model.evaluate(valid_dataset, [dc.metrics.pearson_r2_score])
            
            metrics = {
                'train_r2': train_scores['pearson_r2_score'],
                'valid_r2': valid_scores['pearson_r2_score']
            }
        
        self.trained = True
        return metrics
    
    def predict(self, dataset: dc.data.Dataset) -> np.ndarray:
        """Make predictions."""
        if not self.trained:
            raise ValueError("Model not trained yet.")
        
        if self.model_type == 'rf':
            return self.model.predict(dataset.X)
        else:
            return self.model.predict(dataset)
    
    def predict_single(self, smiles: str, featurizer) -> float:
        """Predict binding affinity for a single SMILES."""
        # Create temporary dataset
        if self.model_type == 'rf':
            features = featurizer.featurize_molecule(smiles)
            if features is None:
                return np.nan
            prediction = self.model.predict(features.reshape(1, -1))
            return prediction[0]
        else:
            temp_dataset = featurizer.create_deepchem_dataset(
                pd.DataFrame({'smiles': [smiles]}), target_column=None
            )
            prediction = self.model.predict(temp_dataset)
            return prediction[0][0]
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance (for tree-based models)."""
        if self.model_type == 'rf' and self.trained:
            return self.model.feature_importances_
        return None

class PsychedelicActivityPredictor:
    """Comprehensive predictor for psychedelic activity."""
    
    def __init__(self):
        self.binding_model = None
        self.activity_model = None
        self.toxicity_model = None
        
    def setup_models(self, tasks: List[str]):
        """Setup all prediction models."""
        # 5-HT2A binding affinity model
        self.binding_model = PsychedelicSARModel('attentivefp')
        self.binding_model.create_model(tasks=['binding_affinity'])
        
        # Psychedelic activity classification
        self.activity_model = PsychedelicSARModel('graphconv')
        self.activity_model.create_model(tasks=['psychedelic_activity'], mode='classification')
        
        # Toxicity prediction
        self.toxicity_model = PsychedelicSARModel('multitask')
        self.toxicity_model.create_model(tasks=['cardiotoxicity', 'neurotoxicity'], n_tasks=2)
    
    def train_all_models(self, datasets: Dict[str, dc.data.Dataset], nb_epoch: int = 100):
        """Train all models."""
        results = {}
        
        if 'binding' in datasets and self.binding_model:
            train_ds, valid_ds = self.binding_model.prepare_data(datasets['binding'])
            results['binding'] = self.binding_model.train(train_ds, valid_ds, nb_epoch)
        
        if 'activity' in datasets and self.activity_model:
            train_ds, valid_ds = self.activity_model.prepare_data(datasets['activity'])
            results['activity'] = self.activity_model.train(train_ds, valid_ds, nb_epoch)
        
        if 'toxicity' in datasets and self.toxicity_model:
            train_ds, valid_ds = self.toxicity_model.prepare_data(datasets['toxicity'])
            results['toxicity'] = self.toxicity_model.train(train_ds, valid_ds, nb_epoch)
        
        return results
    
    def comprehensive_prediction(self, smiles: str, featurizer) -> Dict[str, float]:
        """Make comprehensive prediction for a compound."""
        predictions = {}
        
        if self.binding_model and self.binding_model.trained:
            predictions['5ht2a_binding'] = self.binding_model.predict_single(smiles, featurizer)
        
        if self.activity_model and self.activity_model.trained:
            predictions['psychedelic_activity'] = self.activity_model.predict_single(smiles, featurizer)
        
        if self.toxicity_model and self.toxicity_model.trained:
            tox_pred = self.toxicity_model.predict_single(smiles, featurizer)
            if hasattr(tox_pred, '__len__'):
                predictions['cardiotoxicity'] = tox_pred[0]
                predictions['neurotoxicity'] = tox_pred[1] if len(tox_pred) > 1 else 0
            else:
                predictions['toxicity'] = tox_pred
        
        return predictions

class SARAnalyzer:
    """Structure-Activity Relationship analyzer."""
    
    def __init__(self):
        self.model = None
        self.featurizer = None
    
    def setup(self, model: PsychedelicSARModel, featurizer):
        """Setup analyzer with trained model and featurizer."""
        self.model = model
        self.featurizer = featurizer
    
    def analyze_substitution_effects(self, base_smiles: str, 
                                   substitutions: List[Tuple[str, str]]) -> pd.DataFrame:
        """Analyze effects of different substitutions."""
        results = []
        
        # Base compound
        base_prediction = self.model.predict_single(base_smiles, self.featurizer)
        results.append({
            'compound': 'base',
            'smiles': base_smiles,
            'substitution': 'none',
            'predicted_binding': base_prediction
        })
        
        # Substituted compounds
        for substitution_name, substituted_smiles in substitutions:
            prediction = self.model.predict_single(substituted_smiles, self.featurizer)
            results.append({
                'compound': substitution_name,
                'smiles': substituted_smiles,
                'substitution': substitution_name,
                'predicted_binding': prediction
            })
        
        df = pd.DataFrame(results)
        df['binding_change'] = df['predicted_binding'] - base_prediction
        
        return df
    
    def plot_sar_analysis(self, sar_results: pd.DataFrame, save_path: Optional[str] = None):
        """Plot SAR analysis results."""
        plt.figure(figsize=(12, 8))
        
        # Binding affinity plot
        plt.subplot(2, 2, 1)
        sns.barplot(data=sar_results, x='substitution', y='predicted_binding')
        plt.title('Predicted 5-HT2A Binding Affinity')
        plt.xticks(rotation=45)
        
        # Binding change plot
        plt.subplot(2, 2, 2)
        sns.barplot(data=sar_results, x='substitution', y='binding_change')
        plt.title('Change in Binding Affinity')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def create_sar_model(model_type: str = 'attentivefp') -> PsychedelicSARModel:
    """Factory function to create SAR model."""
    return PsychedelicSARModel(model_type)

if __name__ == "__main__":
    # Test model creation
    model = create_sar_model('attentivefp')
    model.create_model(['binding_affinity'])
    print(f"Created {model.model_type} model for SAR prediction")