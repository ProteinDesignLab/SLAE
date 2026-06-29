"""Build the encoder and decoder from the packaged configs (no checkpoint)."""


def test_default_configs_load():
    from SLAE.inference import default_decoder_config, default_encoder_config

    enc_cfg = default_encoder_config()
    dec_cfg = default_decoder_config()

    assert isinstance(enc_cfg, dict) and enc_cfg
    assert isinstance(dec_cfg, dict) and dec_cfg
    # `_target_` is a Hydra key and must be stripped before passing as kwargs.
    assert "_target_" not in enc_cfg
    assert "_target_" not in dec_cfg


def test_build_models_from_packaged_configs():
    from SLAE.inference import default_decoder_config, default_encoder_config
    from SLAE.model.decoder import AllAtomDecoder
    from SLAE.model.encoder import ProteinEncoder

    encoder = ProteinEncoder(**default_encoder_config())
    decoder = AllAtomDecoder(**default_decoder_config())

    assert isinstance(encoder, ProteinEncoder)
    assert isinstance(decoder, AllAtomDecoder)
