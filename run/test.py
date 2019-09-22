import tensorflow as tf

def valid_step(model, loss_function, encoder_input, decoder_input, target):
    print(encoder_input.shape, decoder_input.shape, target.shape)
    logits = model(inputs = (encoder_input, decoder_input), 
                    train = True)
    loss_value = loss_function(target, logits)
    
    return loss_value