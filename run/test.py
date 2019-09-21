import tensorflow as tf

def test_step(model, loss_function, encoder_input, target):
    logits = model(inputs = (encoder_input, None), 
                    train = False)
    loss_value = loss_function(target, logits)
    
    return loss_value