import tensorflow as tf

class Attention(tf.keras.Model):
    def __init__(self, hidden_size, num_heads):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.q_layer = tf.keras.layers.Dense(units = self.hidden_size,
                                        activation = None,
                                        use_bias = False)
        self.k_layer = tf.keras.layers.Dense(units = self.hidden_size,
                                        activation = None,
                                        use_bias = False)
        self.v_layer = tf.keras.layers.Dense(units = self.hidden_size,
                                        activation = None,
                                        use_bias = False)

        self.output_dende = tf.keras.layers.Dense(units = self.hidden_size,
                                        activation = None,
                                        use_bias = False)

    def split_head(self, x):
        batch_size = x.shape[0]
        length = x.shape[1]
        depth = self.hidden_size//self.num_heads

        x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

        return tf.transpose(x, [0,2,1,3])

    def combine_head(self, x):
        batch_size = x.shape[0]
        length = x.shape[2]
        x = tf.transpose(x, [0,2,1,3])
        return tf.reshape(x, [batch_size, length, self.hidden_size])
    
    def call(self, inputs, bias):
        x,y = inputs
        q = self.q_layer(x)
        k = self.k_layer(y)
        v = self.v_layer(y)

        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)


        depth = self.hidden_size//self.num_heads

        q *= depth**0.5

        logits = tf.matmul(q, k, transpose_b = True)
        logits += bias
        weights = tf.nn.softmax(logits)

        attention_output = tf.matmul(weights, v)
        attention_output = self.combine_head(attention_output)

        attention_output = self.output_dende(attention_output)

        return attention_output

class Self_attention(Attention):
    def call(self, x, bias):
        return super(Self_attention, self).call((x,x),bias)