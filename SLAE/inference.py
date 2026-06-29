"""
High-level inference API for the SLAE autoencoder.

This module wraps the encoder/decoder loading and batching boilerplate so that
loading a pretrained checkpoint and running encode / encode-decode takes a few
lines. It does not change any model behavior -- it simply consolidates the logic
that previously lived in the example scripts.

Typical usage::

    import torch
    from SLAE import load_autoencoder, encode, encode_decode
    from SLAE.features.graph_featurizer import ProteinGraphFeaturizer
    from SLAE.datasets.datamodule import PDBDataModule

    encoder, decoder = load_autoencoder("checkpoints/autoencoder_pdb.ckpt")

    dm = PDBDataModule(pdb_dir="pdbs", processed_dir="processed",
                       inference_only=True, batch_size=1)
    dm.setup(stage="lazy_init")
    featurizer = ProteinGraphFeaturizer(radius=8.0, use_atom37=True)

    batch = next(iter(dm.inference_dataloader()))
    batch = featurizer(batch.to(encoder_device))

    embeddings = encode(encoder, batch)
    result = encode_decode(encoder, decoder, batch)
"""

from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import yaml

from SLAE.model.decoder import AllAtomDecoder
from SLAE.model.encoder import ProteinEncoder
from SLAE.util.constants import FILL

__all__ = [
    "load_autoencoder",
    "encode",
    "encode_decode",
    "default_encoder_config",
    "default_decoder_config",
]

PathLike = Union[str, Path]


def _load_yaml(path: PathLike) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML at {path} did not parse to a dict.")
    return cfg


def default_encoder_config() -> dict:
    """Return the packaged encoder config as a plain kwargs dict."""
    with resources.as_file(
        resources.files("SLAE").joinpath("configs/encoder/protein_encoder.yaml")
    ) as p:
        cfg = _load_yaml(p)
    cfg.pop("_target_", None)
    return cfg


def default_decoder_config() -> dict:
    """Return the packaged all-atom decoder config as a plain kwargs dict."""
    with resources.as_file(
        resources.files("SLAE").joinpath("configs/decoder/allatom_decoder.yaml")
    ) as p:
        cfg = _load_yaml(p)["allatom_decoder"]
    cfg.pop("_target_", None)
    return cfg


def _split_checkpoint(ckpt: dict) -> Tuple[dict, dict]:
    """Extract encoder/decoder state dicts from either supported layout.

    Two checkpoint layouts are supported:

    1. ``{"encoder": {...}, "decoder": {...}}`` -- clean component state dicts.
    2. ``{"state_dict": {...}}`` -- a Lightning checkpoint where keys are
       prefixed with ``encoder.`` and ``decoder.allatom_decoder.``.
    """
    if "encoder" in ckpt and "decoder" in ckpt:
        return ckpt["encoder"], ckpt["decoder"]

    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        enc_state = {
            k[len("encoder."):]: v
            for k, v in sd.items()
            if k.startswith("encoder.")
        }
        dec_state = {
            k[len("decoder.allatom_decoder."):]: v
            for k, v in sd.items()
            if k.startswith("decoder.allatom_decoder.")
        }
        return enc_state, dec_state

    raise KeyError(
        "Unrecognized checkpoint format: expected top-level 'encoder'/'decoder' "
        "keys or a 'state_dict' with 'encoder.'/'decoder.allatom_decoder.' "
        f"prefixes. Found keys: {list(ckpt.keys())}"
    )


def load_autoencoder(
    ckpt_path: PathLike,
    encoder_config: Optional[dict] = None,
    decoder_config: Optional[dict] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = False,
) -> Tuple[ProteinEncoder, AllAtomDecoder]:
    """Build the encoder/decoder and load weights from a checkpoint.

    :param ckpt_path: Path to the ``.ckpt`` file (see Checkpoints in the README).
    :param encoder_config: Encoder kwargs. Defaults to the packaged config.
    :param decoder_config: Decoder kwargs. Defaults to the packaged config.
    :param device: Device to place the models on. Defaults to CUDA if available.
    :param strict: Passed to ``load_state_dict``. ``False`` by default because
        the encoder uses lazy layers that materialize on the first forward pass.
    :returns: ``(encoder, decoder)``, both in ``eval()`` mode on ``device``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    enc_cfg = encoder_config if encoder_config is not None else default_encoder_config()
    dec_cfg = decoder_config if decoder_config is not None else default_decoder_config()

    encoder = ProteinEncoder(**enc_cfg).to(device).eval()
    decoder = AllAtomDecoder(**dec_cfg).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_state, dec_state = _split_checkpoint(ckpt)

    encoder.load_state_dict(enc_state, strict=strict)
    decoder.load_state_dict(dec_state, strict=strict)

    return encoder, decoder


@torch.no_grad()
def encode(encoder: ProteinEncoder, batch) -> Dict[str, torch.Tensor]:
    """Run the encoder on a featurized batch.

    :param encoder: A :class:`ProteinEncoder`.
    :param batch: A featurized PyG batch (see ``ProteinGraphFeaturizer``).
    :returns: Dict with ``node_embedding``, ``residue_embedding`` and
        ``graph_embedding``.
    """
    return encoder(batch)


@torch.no_grad()
def encode_decode(
    encoder: ProteinEncoder,
    decoder: AllAtomDecoder,
    batch,
) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
    """Encode a batch and decode it back to all-atom coordinates.

    :param encoder: A :class:`ProteinEncoder`.
    :param decoder: An :class:`AllAtomDecoder`.
    :param batch: A featurized PyG batch.
    :returns: Dict with:

        - ``backbone_coords``  -- ``[B, N, 4, 3]``
        - ``sidechain_coords`` -- ``[B, N, 33, 3]``
        - ``atom_mask``        -- ``[B, N, 37]`` (bool)
        - ``allatom_coords``   -- ``[B, N, 37, 3]``, invalid atoms set to ``FILL``
        - ``coords_per_sample``-- list of ``[N_i, 37, 3]`` tensors (CPU), each
          cropped to its residue count and masked, ready for ``to_pdb``.
    """
    enc_out = encoder(batch)
    res_embedding = enc_out["residue_embedding"]  # [sum L_i, D]

    residue_batch = batch.residue_batch  # [sum L_i]
    batch_size = int(residue_batch.max().item()) + 1
    lengths = torch.bincount(residue_batch, minlength=batch_size)  # [B]

    # Split per-sample then right-pad to [B, max_len, D].
    seq_list, start = [], 0
    for length in lengths:
        n = int(length.item())
        seq_list.append(res_embedding[start:start + n])
        start += n

    res_embedding_padded = torch.nn.utils.rnn.pad_sequence(
        seq_list, batch_first=True, padding_value=FILL
    )
    max_len = res_embedding_padded.size(1)
    mask = (
        torch.arange(max_len, device=res_embedding_padded.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )

    decoder_output = decoder(
        res_embedding_padded,
        mask,
        batch=batch.residue_batch,
        res_idx=batch.residue_id,
    )

    bb = decoder_output["backbone_coords"]            # [B, N, 4, 3]
    sc = decoder_output["sidechain_coords"]           # [B, N, 33, 3]
    atom_mask = decoder_output["atom_mask"].bool()    # [B, N, 37]
    allatom = torch.cat([bb, sc], dim=2)              # [B, N, 37, 3]

    # Per-sample coordinates, cropped to residue count and masked to FILL.
    coords_per_sample: List[torch.Tensor] = []
    for i in range(allatom.size(0)):
        n = int(lengths[i].item())
        coords = allatom[i, :n].detach().cpu()                  # [N_i, 37, 3]
        m = atom_mask[i, :n].detach().cpu().unsqueeze(-1)       # [N_i, 37, 1]
        coords = torch.where(m, coords, torch.full_like(coords, FILL))
        coords_per_sample.append(coords)

    return {
        "backbone_coords": bb,
        "sidechain_coords": sc,
        "atom_mask": atom_mask,
        "allatom_coords": allatom,
        "coords_per_sample": coords_per_sample,
    }
