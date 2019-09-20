class Embedding(object):
    def __init__(self, vocab_size, output_dim):
        super(Embedding, self).__init__()
        self.W = tf.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[V, d]))
    
    def call(self, word_indexes):
        return tf.nn.embedding_lookup(self.W, word_indexes)