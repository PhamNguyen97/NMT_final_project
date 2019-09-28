from model.encoder import Encoder
# from model.decoder import Decoder
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, encoder_cfg = None, decoder_cfg = None, with_att = False):
        super(Model, self).__init__()
        self.encoder = Encoder(**encoder_cfg)
        if with_att:
            from model.decoder_att import Decoder
        else:
            from model.decoder import Decoder
        self.decoder = Decoder(**decoder_cfg)
    
    def call(self, inputs, train = True):
        encoder_input, decoder_input = inputs
        encoder_states, encoder_last_states = self.encoder(inputs = encoder_input)
        output = self.decoder(inputs = decoder_input, 
                            encoder_hidden_state = encoder_last_states, 
                            encoder_states = encoder_states,
                            train = train)
        return output