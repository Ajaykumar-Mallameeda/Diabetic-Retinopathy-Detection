"""Preprocessing module for retinal image processing"""

try:
    from .classical_filters import *
except ImportError:
    pass
try:
    from .wavelet_transforms import *
except ImportError:
    # PyWavelets might not be installed
    pass