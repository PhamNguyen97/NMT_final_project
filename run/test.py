import tensorflow as tf
from run.bleu import compute_bleu

def valid_step(model, loss_function, encoder_input, decoder_input, target):
    logits = model(inputs = (encoder_input, decoder_input), 
                    train = True)
    loss_value = loss_function(target, logits)
    
    return loss_value

def test_step(model, encoder_input, target):
    pass

