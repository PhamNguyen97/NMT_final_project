import tensorflow as tf
from preprocess_data.word2vec import Data_processing
from random import shuffle

class Data_loader(object):
    def __init__(self, 
                vi_train = 'nmt_data/vie-eng-iwslt/train.vi',
                eng_train = 'nmt_data/vie-eng-iwslt/train.en',
                vi_test = 'nmt_data/vie-eng-iwslt/tst2012.vi',
                eng_test = 'nmt_data/vie-eng-iwslt/tst2012.en',
                mode = 'test',
                batch_size = 5,
                max_length = 40
                ):
        self.data_processor = Data_processing(vi_train = vi_train,
                                            eng_train = eng_train,
                                            mode = mode,
                                            max_length = max_length)

        self.source_train = open(eng_train, 'r').readlines()
        self.target_train = open(vi_train, 'r').readlines()

        self.source_test = open(eng_test, 'r').readlines()
        self.target_test = open(vi_test, 'r').readlines()

        self.data_ids = list(range(len(self.source_train)))
        shuffle(self.data_ids)

        self.valid_data_ids = self.data_ids[:int(len(self.data_ids)*0.2)].copy()
        self.data_ids = self.data_ids[int(len(self.data_ids)*0.2)::].copy()

        self.test_data_ids = list(range(len(self.source_test)))

        self.dataset = tf.data.Dataset.from_generator(
            generator = self.generator,
            output_types = (tf.int64, tf.int64, tf.int64)
        )
        self.dataset = self.dataset.batch(batch_size, drop_remainder = True)
        self.num_step = len(self.data_ids)//batch_size

        self.valid_dataset = tf.data.Dataset.from_generator(
            generator = self.valid_generator,
            output_types = (tf.int64, tf.int64, tf.int64)
        )
        self.num_valid_step = len(self.valid_data_ids)//batch_size


        self.test_dataset = tf.data.Dataset.from_generator(
            generator = self.test_generator,
            output_types = (tf.int64, tf.int64, tf.int64)
        )
        self.test_dataset = self.test_dataset.batch(batch_size, drop_remainder = True)
        self.num_test_step = len(self.test_data_ids)//batch_size

    def generator(self):
        for index, data_id in enumerate(self.data_ids):
            source_vec = self.data_processor(sentence = self.source_train[data_id], 
                                            vi = False, 
                                            to_id = True)
            target_vec = self.data_processor(sentence = self.target_train[data_id],
                                            vi = True,
                                            to_id = True)
            if index==len(self.data_ids):
                shuffle(self.data_ids)
                print('shuffled data')
            yield (source_vec, target_vec[:-1], target_vec[1:])
    
    def valid_generator(self):
        for index, data_id in enumerate(self.valid_data_ids):
            source_vec = self.data_processor(sentence = self.source_train[data_id], 
                                            vi = False, 
                                            to_id = True)
            target_vec = self.data_processor(sentence = self.target_train[data_id],
                                            vi = True,
                                            to_id = True)

            yield (source_vec, target_vec[:-1], target_vec[1:])
    
    def test_generator(self):
        for index, data_id in enumerate(self.test_data_ids):
            source_vec = self.data_processor(sentence = self.source_test[data_id], 
                                            vi = False, 
                                            to_id = True)
            target_vec = self.data_processor(sentence = self.target_test[data_id],
                                            vi = True,
                                            to_id = True)
           
            yield (source_vec, target_vec[:-1], target_vec[1:])
    

    
