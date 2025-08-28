"""
Psychedelic Therapeutics Design Framework
=========================================

A comprehensive toolkit for designing novel psychedelic therapeutics 
targeting the 5-HT2A receptor, with focus on 2C-B and its analogs.

Modules:
--------
- data: Data collection and curation for 2C-series compounds
- molecular: Molecular featurization and representation
- models: SAR prediction models and graph neural networks  
- structural: Binding pocket analysis and allosteric site identification
- generation: Novel compound generation and optimization
- analysis: Visualization and analysis tools

Authors: CNS Therapeutics Research Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "CNS Therapeutics Research Team"

# Core imports
from . import data
from . import molecular
from . import models
from . import structural
from . import generation
from . import analysis

__all__ = [
    'data',
    'molecular', 
    'models',
    'structural',
    'generation',
    'analysis'
]