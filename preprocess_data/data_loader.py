import tensorflow as tf
from preprocess_data.word2vec import Data_processing
from random import shuffle

class Data_loader(object):
    def __init__(self, 
                vi_train = 'nmt_data/vie-eng-iwslt/train.vi',
                eng_train = 'nmt_data/vie-eng-iwslt/train.en',
                mode = 'test',
                batch_size = 5,
                max_length = 40
                ):
        self.data_processor = Data_processing(vi_train = vi_train,
                                            eng_train = eng_train,
                                            mode = mode,
                                            max_length = max_length)
        self.source_train = open(vi_train, 'r').readlines()
        self.target_train = open(eng_train, 'r').readlines()
        self.data_ids = list(range(len(self.source_train)))
        shuffle(self.data_ids)

        self.dataset = tf.data.Dataset.from_generator(
            generator = self.generator,
            output_types = (tf.int64, tf.int64, tf.int64)
        )
        self.dataset = self.dataset.batch(batch_size, drop_remainder = True)

    def generator(self):
        for index, data_id in enumerate(self.data_ids):
            source_vec = self.data_processor(sentence = self.source_train[data_id], 
                                            vi = True, 
                                            to_id = True)
            target_vec = self.data_processor(sentence = self.target_train[data_id],
                                            vi = False,
                                            to_id = True)
            if index==len(self.data_ids):
                shuffle(self.data_ids)
                print('shuffled data')
            
            yield (source_vec, target_vec[:-1], target_vec[1:])
    

    
