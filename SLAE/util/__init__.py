# ruff: noqa: I001
# NOTE: import order here is deliberate and must not be auto-sorted.
# `_irrep_utils` must be imported before `SLAE.nn.*`, because importing
# `SLAE.nn.base_model` pulls in `SLAE.nn.graph_mixin`, which does
# `from SLAE.util import _fix_irreps_dict, _irreps_compatible`. Those names must
# already be bound on this module to avoid a circular-import error.
"""
Utility functions and helpers for SLAE.

This module provides various utility functions including:
- types: Type definitions and data structures
- constants: Protein structure constants (atom37 ordering, etc.)
- avg_neighbor: Average-neighbor-count estimation for the encoder
"""

from SLAE.util._irrep_utils import _fix_irreps_dict, _irreps_compatible
from SLAE.util.types import ModelOutput
from SLAE.nn.base_model import BaseModel
from SLAE.nn.mlp_decoder import MLPDecoder, PositionDecoder

__all__ = [
    # Irrep utilities
    "_fix_irreps_dict",
    "_irreps_compatible",

    # Types
    "ModelOutput",

    # Base classes
    "BaseModel",

    # Decoder building blocks
    "MLPDecoder",
    "PositionDecoder",
]
