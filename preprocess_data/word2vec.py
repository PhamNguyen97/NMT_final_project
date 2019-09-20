import re
import numpy as np
import tensorflow as tf

def sentence_to_words(sentence):
    sentence = re.sub('\d+|[.,!?;]',' ', sentence.lower()).strip()
    return re.findall(r"[\w']+|[.,!?;]", sentence)

def make_vocab(data_file, thresh = 100, mode = 'test'):
    words = dict()
    with open(data_file, 'r', encoding="utf8") as file:
        for line_index, line in enumerate(file):
            if mode=='test' and line_index == 10:
                thresh = 1
                break
            for item in sentence_to_words(line):
                if not item in words:
                    words[item] = 1
                else:
                    words[item] +=1
    
    words = dict(filter(lambda item: item[1]>=thresh, words.items()))
    ids = dict()
    for index, key in enumerate(words):
        words[key] = index+3
        ids[str(index+3)] = key
    words['<bos>'] = 1
    words['<eos>'] = 2
    words['<unk>'] = 0

    ids['1'] = '<bos>'
    ids['2'] = '<eos>'
    num_words = index+4
    return words, ids, num_words

def sentence_to_ids(sentence, vocab, max_length):
    line_words = sentence_to_words(sentence)
    line_words = list(map(lambda word: vocab[word] if word in vocab else vocab['<unk>'] , line_words))
    line_words = list(filter(lambda word_id: word_id!= 0, line_words))
    if len(line_words) > max_length-2:
        line_words = line_words[:-2]
    line_words = [1, *line_words, 2]
    line_words = line_words+[0]*(max_length-len(line_words))
    return line_words

def ids_to_sentence(ids, vocab_ids):
    return list(map(lambda id: vocab_ids[str(id)], ids))


class Data_processing(object):
    def __init__(self, vi_train, eng_train, mode, max_length):
        self.viet_vocab, self.viet_ids, self.viet_size = make_vocab(vi_train, mode = mode)
        self.eng_vocab, self.eng_ids, self.eng_size = make_vocab(eng_train, mode = mode)
        self.max_length = max_length
    def __call__(self, sentence, vi = True, to_id = True):
        if vi:
            if to_id:
                return np.array(sentence_to_ids(sentence, self.viet_vocab, self.max_length))
            else:
                return ids_to_sentence(sentence, self.viet_ids)
        else:
            if to_id:
                return np.array(sentence_to_ids(sentence, self.eng_vocab, self.max_length))
            else:
                return ids_to_sentence(sentence, self.eng_ids)


