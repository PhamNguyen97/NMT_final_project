import tensorflow as tf 

class Loss(object):
    def __init__(self):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                                from_logits=True, 
                                reduction='none')
        
    def __call__(self, target, logits):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = self.loss_object(target, logits)

        mask = tf.cast(mask, dtype = loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)


def train_step(model, loss_function, optimizer, encoder_input, decoder_input, target):
    with tf.GradientTape() as tape:
        logits = model(encoder_input = encoder_input, 
                        decoder_input = decoder_input, 
                        train = True)
        loss_value = loss_function(target, logits)
    
    model_trainable_variables = model.get_trainable_variables()
    grads = tape.gradient(loss_value, model_trainable_variables)
    optimizer.apply_gradients(zip(grads, model_trainable_variables))

    return model, loss_value


