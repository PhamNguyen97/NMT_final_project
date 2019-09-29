from model.attention import Attention
import tensorflow as tf
import numpy as np
class Decoder(tf.keras.Model):
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
                num_lstm_layer = 1,
                max_length = 40,
                fully_connected_cfg={
                    "units": -1,
                    "activation":None,
                    "use_bias":True
                },
                attention_cfg = {
                    "hidden_size": 64, 
                    "num_heads":1
                }):
        super(Decoder, self).__init__()

        self.max_length = 40
        self.embedding = tf.keras.layers.Embedding(**embedding_cfg)    
        self.lstm_layers = []
        for _ in range(num_lstm_layer):
            self.lstm_layers.append(tf.keras.layers.LSTM(**LSTM_cfg))
        
        self.attention = Attention(**attention_cfg)
        self.fully_connected = tf.keras.layers.Dense(**fully_connected_cfg)

    
    def call(self, inputs, encoder_hidden_state, encoder_states, train = False):
        if train:
            initial_states = encoder_hidden_state
            all_states = []
            batch_size = initial_states[0][0].shape[0]
            output = []
            for current_word_index in range(inputs.shape[1]):
                current_word = inputs[:,current_word_index]
                current_word = tf.expand_dims(current_word, 1)
                all_state = self.embedding(current_word)
                current_initial_state = []
                for lstm_layer, initial_state, ecoder_state in zip(self.lstm_layers, initial_states, encoder_states):
                    context_vec = self.attention((tf.expand_dims(initial_state[0], 1),encoder_state), 0)  
                    all_state = tf.concate([all_state, context_vec])
                    ####### fix this
                    all_state, h, c = lstm_layer(all_state, initial_state = initial_state)
                    current_initial_state.append((h,c))
                context_vec = self.attention((tf.expand_dims(h, 1),encoder_states), 0)
                all_state = tf.concat([context_vec, all_state], 2)    
                current_word = self.fully_connected(all_state)
                output.append(current_word)  

                initial_states = current_initial_state
            output = tf.concat(output,1)
            return output
        else:
            initial_states = encoder_hidden_state
            all_states = []
            batch_size = initial_states[0][0].shape[0]
            current_word = np.ones((batch_size,1))
            output = []
            for _ in range(self.max_length-1):
                all_state = self.embedding(current_word)
                current_initial_state = []
                for lstm_layer, initial_state in zip(self.lstm_layers, initial_states):
                    all_state, h, c = lstm_layer(all_state, initial_state = initial_state)
                    current_initial_state.append((h,c))
                context_vec = self.attention((tf.expand_dims(h, 1),encoder_states), 0)
                all_state = tf.concat([context_vec, all_state], 2)    

                current_word = self.fully_connected(all_state)
                current_word = tf.argmax(current_word, axis = 2)
                output.append(current_word)  

                initial_states = current_initial_state
            output = tf.concat(output,1)
            return output

