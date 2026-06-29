#!/usr/bin/env python
"""Encode protein structures into SLAE embeddings.

Example::

    python scripts/encode_example.py \
        --ckpt checkpoints/autoencoder_pdb.ckpt \
        --pdb-dir data/pdb_store \
        --processed-dir data/processed
"""

import argparse

import torch

from SLAE import encode, load_autoencoder
from SLAE.datasets.datamodule import PDBDataModule
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="Path to the autoencoder .ckpt")
    p.add_argument("--pdb-dir", required=True, help="Directory of input PDB files")
    p.add_argument(
        "--processed-dir", required=True, help="Directory for processed graph cache"
    )
    p.add_argument("--radius", type=float, default=8.0, help="Graph cutoff radius (A)")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[device] {device}")

    encoder, _ = load_autoencoder(args.ckpt, device=device)

    dm = PDBDataModule(
        pdb_dir=args.pdb_dir,
        processed_dir=args.processed_dir,
        inference_only=True,
        batch_size=1,
    )
    dm.setup(stage="lazy_init")
    featurizer = ProteinGraphFeaturizer(radius=args.radius, use_atom37=True)

    batch = next(iter(dm.inference_dataloader())).to(device)
    batch = featurizer(batch)

    embeddings = encode(encoder, batch)
    residue_embeddings = embeddings["residue_embedding"].cpu()  # (N_residues, 128)

    print(f"[id] {batch.id[0]}")
    print(f"[residue_embedding] {tuple(residue_embeddings.shape)}")
    # Also available: embeddings["node_embedding"], embeddings["graph_embedding"].


if __name__ == "__main__":
    main()
