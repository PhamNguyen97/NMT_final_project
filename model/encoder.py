import tensorflow as tf 
tf.compat.v1.enable_eager_execution()

class Encoder(tf.keras.Model):
    def __init__(self, 
                embedding_cfg = {        
                    "input_dim": -1,
                    "output_dim": 64,
                    "embeddings_initializer":"uniform",
                    "embeddings_regularizer": "l2",
                    "activity_regularizer": "l2",
                    "mask_zero": True,
                    "input_length": 40
                },
                LSTM_cfg = {
                    "units": 64,
                    "activation":"tanh",
                    "recurrent_activation":"sigmoid",
                    "use_bias":True,
                    "kernel_initializer":"glorot_uniform",
                    "bias_initializer":"zeros",
                    "unit_forget_bias":True,
                    "kernel_regularizer":"l2",
                    "recurrent_regularizer":"l2",
                    "bias_regularizer":"l2",
                    "activity_regularizer":"l2",
                    "dropout":0.1,
                    "recurrent_dropout":0.1,
                    "implementation":2,
                    "return_sequences":True,
                    "return_state":True,
                    "go_backwards":False,
                    "stateful":False,
                    "unroll":False
                },
                num_lstm_layer = 1):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(**embedding_cfg)
        self.lstm_layers = []
        for _ in range(num_lstm_layer):
            self.lstm_layers.append(tf.keras.layers.LSTM(**LSTM_cfg))

    def call(self, inputs, return_all_states = False):
        all_state = self.embedding(inputs)
        last_states = [],
        all_states = []
        for lstm_layer in self.lstm_layers:
            all_state, h, c = lstm_layer(all_state)
            last_states.append((h,c))
            if return_all_states:
                all_states.append(all_state)
        
        if return_all_states:
            return all_state, last_states, all_states
        else:
            return all_state, last_states
        
        
