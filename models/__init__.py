# models/__init__.py

from .adahgnn import EEG_AdaMSHyper
from .layers import LearnableTemporalShift, GradientReversalLayer
from .hypergraph import MultiScale_EEG_Hypergraph

# 定义当使用 from models import * 时导出的内容
__all__ = [
    'EEG_AdaMSHyper',
    'LearnableTemporalShift',
    'GradientReversalLayer',
    'MultiScale_EEG_Hypergraph'
]