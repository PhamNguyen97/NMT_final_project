import argparse
import json
from model.encoder_decoder_arc import Model
from preprocess_data.data_loader import Data_loader
from run.train import train_step, Loss
from run.test import valid_step
import tensorflow as tf
import sys
import warnings
import copy
import os
import time
tf.compat.v1.enable_eager_execution()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configure', help = 'configure file')
    parser.add_argument('-m', '--mode', help = 'real run or test mode')
    parser.add_argument('-b', '--batch_size', default=None, help = 'batch size')
    parser.add_argument('-cp', '--checkpoint_dir', default=None, help = 'checkpoint folder')
    parser.add_argument('-rs', '--resume', default=None, help = 'resume checkpoint from checkpoint_dir')

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
    if args.checkpoint_dir is None:
        checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    else:
        checkpoint_dir = args.checkpoint_dir
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    if args.resume is not None:
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print('resumed checkpoint :', status)
    # training loop
    with tf.device('/device:GPU:0'):
        for epoch in range(num_epochs):
            total_train_loss = 0
            start_time = time.time()

            for index, (eng_inp, vi_inp, vi_tar) in enumerate(data_loader.dataset):
                
                model, step_loss = train_step(model = model, 
                                    loss_function = loss_function, 
                                    optimizer = optimizer, 
                                    encoder_input = eng_inp, 
                                    decoder_input = vi_inp, 
                                    target = vi_tar)
                total_train_loss+=step_loss
                if (index+1)%num_step_to_print ==0:
                    total_valid_loss = 0
                    for _, (test_eng_inp, test_vi_inp, test_vi_tar) in enumerate(data_loader.test_dataset):
                        step_loss = valid_step(model = model, 
                                    loss_function = loss_function, 
                                    encoder_input = test_eng_inp, 
                                    target = test_vi_tar)
                        total_valid_loss+= step_loss
                    print("Validation_ epoch: {}/{}, loss:{}".format(epoch, num_epochs, total_valid_loss/data_loader.num_test_step))
                    checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "{}_{}_{}.ckpt".format(epoch, index, total_valid_loss/data_loader.num_test_step)))
                    delta_time = time.time()-start_time
                    start_time = time.time()

                    print("epoch: {}/{}, step: {}/{}, loss:{}, time_per_{}_step:{}".format(epoch, 
                                                                                num_epochs,
                                                                                index, 
                                                                                data_loader.num_step, 
                                                                                total_train_loss/num_step_to_print,
                                                                                num_step_to_print,
                                                                                delta_time))
                    total_train_loss = 0
                    print('save checkpoint to:', os.path.join(checkpoint_dir, "{}_{}_{}.ckpt".format(epoch, index, total_valid_loss/data_loader.num_test_step)))





        
if __name__ == "__main__":
    main() 

    


