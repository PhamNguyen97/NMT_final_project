import argparse
import json
from model.encoder_decoder_arc import Model
from preprocess_data.data_loader import Data_loader
from run.train import train_step, Loss
from run.test import test_step
import tensorflow as tf
import sys
import warnings
import copy
import os
tf.compat.v1.enable_eager_execution()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configure', help = 'configure file')
    parser.add_argument('-m', '--mode', help = 'real run or test mode')
    parser.add_argument('-b', '--batch_size', default=None, help = 'batch size')

    args = parser.parse_args()
    mode = args.mode
    if not mode in ['train', 'test']:
        warnings.warn('mode should be "train" or "test"')
        sys.exit()

    config = json.load(open(args.configure, 'r'))
    if args.batch_size is not None:
        config['data_loader_cfg']['batch_size'] = int(args.batch_size)
    num_epochs = config.get('num_epochs', 100)
    num_step_to_print = config.get('num_step_to_print', 100)

    # data loader
    data_loader = Data_loader(**config.get('data_loader_cfg'), mode = mode)
    print('num_train_data:',len(data_loader.data_ids))
    print('num_train_step:',data_loader.num_step)
    print('num_test_data:', len(data_loader.test_data_ids))
    print('num_test_step:', data_loader.num_test_step)

    # define model
    config['encoder_cfg']['embedding_cfg']['input_dim'] = data_loader.data_processor.eng_size
    config['decoder_cfg']['embedding_cfg']['input_dim'] = data_loader.data_processor.viet_size
    config['decoder_cfg']['fully_connected_cfg']['units'] = data_loader.data_processor.viet_size

    model = Model(encoder_cfg = config.get('encoder_cfg'),
                decoder_cfg = config.get('decoder_cfg'))
    
    # define loss function and optimizer
    loss_function = Loss()
    optimizer = tf.keras.optimizers.Adam()

    # checkpoints
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')

    # training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for index, (eng_inp, vi_inp, vi_tar) in enumerate(data_loader.dataset):
            model, step_loss = train_step(model = model, 
                                loss_function = loss_function, 
                                optimizer = optimizer, 
                                encoder_input = eng_inp, 
                                decoder_input = vi_inp, 
                                target = vi_tar)
            total_loss+=step_loss
            if (index+1)%num_step_to_print ==0:
                print("epoch: {}, step: {}/{}, loss:{}".format(epoch, index, data_loader.num_step, total_loss/num_step_to_print))
                total_loss = 0
                # for _, (test_eng_inp, test_vi_inp, test_vi_tar) in enumerate(data_loader.test_dataset):
                #     step_loss = test_step(model = model, 
                #                 loss_function = loss_function, 
                #                 encoder_input = test_eng_inp, 
                #                 target = test_vi_tar)
                #     total_loss+= step_loss
                #     print("Validation_ epoch: {}, loss:{}".format(epoch, total_loss/data_loader.num_test_step))
                #     total_loss = 0
                root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model,
                           optimizer_step=tf.compat.v1.train.get_or_create_global_step())
                root.save(os.path.join(checkpoint_dir, "{}_{}.ckpt".format(epoch, index)))





        
if __name__ == "__main__":
    main() 

    


