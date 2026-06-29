"""
Model architectures for SLAE.

This module contains the core neural network architectures:
- encoder: SLAE protein encoder
- decoder: All-atom decoder
"""

from SLAE.model.decoder import AllAtomDecoder
from SLAE.model.encoder import ProteinEncoder

__all__ = [
    "ProteinEncoder",
    "AllAtomDecoder",
]
