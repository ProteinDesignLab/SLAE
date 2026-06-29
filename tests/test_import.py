"""Smoke test: the package imports and exposes its public API."""


def test_import_and_public_api():
    import SLAE

    assert SLAE.__version__

    expected = {
        "ProteinEncoder",
        "AllAtomDecoder",
        "ProteinDataset",
        "ProteinDataLoader",
        "PDBDataModule",
        "ProteinGraphFeaturizer",
        "transform_representation_fa",
        "load_autoencoder",
        "encode",
        "encode_decode",
    }
    assert expected.issubset(set(SLAE.__all__))
    for name in expected:
        assert hasattr(SLAE, name), f"SLAE.{name} is missing"


def test_no_training_or_downstream_symbols():
    """Training / chemical-shift code is removed from this release."""
    import SLAE

    for name in ("AutoEncoderModel", "ShiftNet", "EmbeddingExtractor"):
        assert not hasattr(SLAE, name), f"{name} should not be exported"
