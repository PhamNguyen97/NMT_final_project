import tensorflow as tf

def test_step(model, loss_function, encoder_input, target):
    logits = model(encoder_input = encoder_input, 
                    decoder_input = None, 
                    train = False)
    loss_value = loss_function(target, logits)
    
    return loss_value