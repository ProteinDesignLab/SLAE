#!/usr/bin/env python
"""Encode a protein structure and decode it back to all-atom coordinates.

Example::

    python scripts/encode_decode_example.py \
        --ckpt checkpoints/autoencoder_pdb.ckpt \
        --pdb-dir data/pdb_store \
        --processed-dir data/processed \
        --out decoded_structure.pdb
"""

import argparse

import torch

from SLAE import encode_decode, load_autoencoder
from SLAE.datasets.datamodule import PDBDataModule
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer
from SLAE.io.write_pdb import to_pdb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="Path to the autoencoder .ckpt")
    p.add_argument("--pdb-dir", required=True, help="Directory of input PDB files")
    p.add_argument(
        "--processed-dir", required=True, help="Directory for processed graph cache"
    )
    p.add_argument(
        "--out", default="decoded_structure.pdb", help="Output PDB path"
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

    encoder, decoder = load_autoencoder(args.ckpt, device=device)

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
    print(f"[id] {batch.id[0]}")

    result = encode_decode(encoder, decoder, batch)

    # First (and only, batch_size=1) sample: [N, 37, 3] with invalid atoms = FILL.
    coords = result["coords_per_sample"][0]
    to_pdb(coords, args.out)
    print(f"[written] {args.out}  ({tuple(coords.shape)})")


if __name__ == "__main__":
    main()
