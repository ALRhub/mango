from mango.simulator.abstract_ml_simulator import AbstractMLSimulator


def get_simulator(simulator_config, train_dl, eval_ds):
    # calculate an example batch to get the sizes of the network inputs and outputs
    example_batch = None
    for example_batch in train_dl:
        break
    assert example_batch is not None, "The train iterator of the environment must not be empty."
    if simulator_config.name == "hmpn_step_simulator":
        from mango.simulator.hmpn_step_simulator import HMPNStepSimulator
        return HMPNStepSimulator(simulator_config, example_batch)
    if "ml" in simulator_config.name:
        # all meta learning simulators
        encoder = get_encoder(simulator_config.encoder, example_batch)
        decoder = get_decoder(simulator_config.decoder, example_batch, eval_ds)
        simulator = AbstractMLSimulator(simulator_config, encoder, decoder)
        return simulator
    else:
        raise NotImplementedError(f"Unknown simulator name: {simulator_config.name}")


def get_encoder(encoder_config, example_batch):
    if encoder_config.name == "egno":
        from mango.simulator.ml_encoder.egno_encoder import EGNOEncoder
        return EGNOEncoder(encoder_config, example_batch)
    elif encoder_config.name == "mgno":
        from mango.simulator.ml_encoder.mgno_encoder import MGNOEncoder
        return MGNOEncoder(encoder_config, example_batch)
    elif encoder_config.name == "mgn":
        from mango.simulator.ml_encoder.mgn_encoder import MGNEncoder
        return MGNEncoder(encoder_config, example_batch)
    elif encoder_config.name == "deepset":
        from mango.simulator.ml_encoder.deepset_encoder import DeepSetEncoder
        return DeepSetEncoder(encoder_config, example_batch)
    elif encoder_config.name == "cnn_deepset":
        from mango.simulator.ml_encoder.cnn_deepset_encoder import CNNDeepSetEncoder
        return CNNDeepSetEncoder(encoder_config, example_batch)
    elif encoder_config.name == "transformer":
        from mango.simulator.ml_encoder.transformer_encoder import TransformerEncoder
        return TransformerEncoder(encoder_config, example_batch)
    elif encoder_config.name == "dummy":
        from mango.simulator.ml_encoder.dummy_encoder import DummyEncoder
        return DummyEncoder(encoder_config, example_batch)
    else:
        raise NotImplementedError(f"Unknown encoder name: {encoder_config.name}")


def get_decoder(decoder_config, example_batch, eval_ds):
    if decoder_config.name == "egno":
        from mango.simulator.ml_decoder.egno_decoder import EGNODecoder
        return EGNODecoder(decoder_config, example_batch)
    elif decoder_config.name == "mgno":
        from mango.simulator.ml_decoder.mgno_decoder import MGNODecoder
        return MGNODecoder(decoder_config, example_batch)
    elif decoder_config.name == "hmpn_step_simulator":
        from mango.simulator.hmpn_step_simulator import HMPNStepSimulator
        return HMPNStepSimulator(decoder_config, example_batch)
    elif decoder_config.name == "egnn":
        from mango.simulator.ml_decoder.egnn_decoder import EGNNDecoder
        return EGNNDecoder(decoder_config, example_batch)
    elif decoder_config.name == "mgn":
        from mango.simulator.ml_decoder.mgn_decoder import MGNDecoder
        return MGNDecoder(decoder_config, example_batch, eval_ds)
    else:
        raise NotImplementedError(f"Unknown decoder name: {decoder_config.name}")
