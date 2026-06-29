"""Checkpoint-dependent test for the inference loader.

Skips when the Git LFS checkpoint has not been pulled (the tracked file is then
just a small LFS pointer, not real weights).
"""

from pathlib import Path

import pytest

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "autoencoder.ckpt"


def _is_real_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False
    # Git LFS pointer files are tiny and start with a version line.
    if path.stat().st_size < 1024:
        return False
    with open(path, "rb") as f:
        head = f.read(64)
    return not head.startswith(b"version https://git-lfs")


@pytest.mark.skipif(
    not _is_real_checkpoint(CKPT),
    reason="checkpoint not pulled (run `git lfs pull`)",
)
def test_load_autoencoder():
    from SLAE.inference import load_autoencoder

    encoder, decoder = load_autoencoder(CKPT, device="cpu")
    assert encoder.training is False
    assert decoder.training is False
