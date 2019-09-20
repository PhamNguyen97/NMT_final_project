import argparse
import json
from model.encoder_decoder_arc import Model
from preprocess_data.data_loader import Data_loader
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configure', help = 'configure file')
    parser.add_argument('-m', '--mode', help = 'real run or test mode')


    args = parser.parse_args()
    config = json.load(open(args.configure, 'r'))
    mode = args.mode

    data_loader = Data_loader(**config.get('data_loader_cfg'), mode = mode)

    config['encoder_cfg']['embedding_cfg']['input_dim'] = data_loader.data_processor.viet_size
    config['decoder_cfg']['embedding_cfg']['input_dim'] = data_loader.data_processor.eng_size

    model = Model(encoder_cfg = config.get('encoder_cfg'),
                decoder_cfg = config.get('decoder_cfg'))
    
    for index, (vi_inp, eng_inp, eng_tar) in enumerate(data_loader.dataset):
        model(encoder_input = vi_inp, decoder_input = eng_inp)
        # print(len(vi_inp))
        break





if __name__ == "__main__":
    main() 

    


