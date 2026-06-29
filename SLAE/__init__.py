"""
SLAE: Strictly Local All-atom Environment for Protein Representation
===================================================================

A deep learning framework for all-atom protein structure representation. This
public release exposes loading the pretrained autoencoder and running inference:

- encode a protein structure into atom / residue / graph embeddings, and
- decode residue embeddings back to all-atom coordinates.

Main components:

- ``model``    : encoder and all-atom decoder architectures
- ``datasets`` : data loading and preprocessing (``PDBDataModule``)
- ``features`` : graph featurization (``ProteinGraphFeaturizer``)
- ``nn``       : neural-network building blocks
- ``util``     : constants and helpers
- ``inference``: high-level ``load_autoencoder`` / ``encode`` / ``encode_decode``
"""

__version__ = "0.1.0"

# Core models
from SLAE.datasets.dataloader import ProteinDataLoader
from SLAE.datasets.datamodule import PDBDataModule

# Data
from SLAE.datasets.dataset import ProteinDataset
from SLAE.features.fa_representation import transform_representation_fa

# Features
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer

# High-level inference API
from SLAE.inference import (
    default_decoder_config,
    default_encoder_config,
    encode,
    encode_decode,
    load_autoencoder,
)
from SLAE.model.decoder import AllAtomDecoder
from SLAE.model.encoder import ProteinEncoder

__all__ = [
    # Models
    "ProteinEncoder",
    "AllAtomDecoder",
    # Data
    "ProteinDataset",
    "ProteinDataLoader",
    "PDBDataModule",
    # Features
    "ProteinGraphFeaturizer",
    "transform_representation_fa",
    # Inference
    "load_autoencoder",
    "encode",
    "encode_decode",
    "default_encoder_config",
    "default_decoder_config",
]
