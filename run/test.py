import tensorflow as tf

def valid_step(model, loss_function, encoder_input, decoder_input, target):
    logits = model(inputs = (encoder_input, decoder_input), 
                    train = True)
    loss_value = loss_function(target, logits)
    
    return loss_value