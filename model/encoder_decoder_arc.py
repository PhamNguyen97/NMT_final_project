from model.encoder import Encoder
from model.decoder import Decoder
import tensorflow as tf
class Model(tf.keras.Model):
    def __init__(self, encoder_cfg = None, decoder_cfg = None):
        super(Model, self).__init__()
        self.encoder = Encoder(**encoder_cfg)
        self.decoder = Decoder(**decoder_cfg)

    def get_trainable_variables(self):
        return [*self.encoder.get_trainable_variables(), *self.decoder.get_trainable_variables()]
    
    def call(self, inputs, train = True, beam_search = False):
        encoder_input, decoder_input = inputs
        encoder_states, encoder_last_states = self.encoder(inputs = encoder_input)
        output = self.decoder(inputs = decoder_input, 
                            encoder_hidden_state = encoder_last_states, 
                            train = train,
                            beam_search = beam_search)
        return output