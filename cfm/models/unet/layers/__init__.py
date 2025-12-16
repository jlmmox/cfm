import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .residual import ResBlock as ResBlock
from .attention import SpatialSelfAttention as SpatialSelfAttention
