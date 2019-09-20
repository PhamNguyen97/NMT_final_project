from model.encoder import Encoder
from model.decoder import Decoder

class Model(object):
    def __init__(self, encoder_cfg = None, decoder_cfg = None):
        self.encoder = Encoder(**encoder_cfg)
        self.decoder = Decoder(**decoder_cfg)

    def __call__(self, encoder_input, decoder_input, train = True):
        encoder_states, encoder_last_states = self.encoder(input = encoder_input)
        decoder_states = self.decoder(input = decoder_input, 
                                    encoder_hidden_state = encoder_last_states, 
                                    train = train)
        print(decoder_states.shape)
        