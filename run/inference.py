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
        print('_______________________]]]]]]]]]]]]]]]', tf.train.latest_checkpoint('./checkpoints/project_1'))
        status = checkpoint.restore('./checkpoints/project_1/model_1.9139268398284912.ckpt-200')
        print('resumed checkpoint :', status)
    # training loop
    with tf.device('/device:GPU:0'):
        for index, (test_eng_inp, test_vi_inp, test_vi_tar) in enumerate(data_loader.test_dataset):
            output = model(inputs = (encoder_input, decoder_input),
                            train = False,
                            beam_search = False)
            print(output)

if __name__ == '__main__':
    main()
        