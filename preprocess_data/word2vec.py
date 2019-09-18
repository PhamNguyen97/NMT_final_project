import re
import matplotlib.pyplot as plt

def sentence_to_words(sentence):
    sentence = re.sub('\d+|[.,!?;]',' ', sentence.lower()).strip()
    return re.findall(r"[\w']+|[.,!?;]", sentence)

def make_vocab(data_file, thresh = 100):
    words = dict()
    with open(data_file, 'r', encoding="utf8") as file:
        for line in file:
            for item in sentence_to_words(line):
                if not item in words:
                    words[item] = 1
                else:
                    words[item] +=1
    
    words = dict(filter(lambda item: item[1]>=thresh, words.items()))
    ids = dict()
    for index, key in enumerate(words):
        words[key] = index+2
        ids[str(index+2)] = key
    words['<bos>'] = 0
    words['<eos>'] = 1
    words['<unk>'] = -1

    ids['0'] = '<bos>'
    ids['1'] = '<eos>'
    num_words = index+3
    return words, ids, num_words

def sentence_to_ids(sentence, vocab):
    line_words = sentence_to_words(sentence)
    line_words = list(map(lambda word: vocab[word] if word in vocab else vocab['<unk>'] , line_words))
    line_words = list(filter(lambda word_id: word_id!= -1, line_words))
    line_words = [0, *line_words, 1]
    return line_words

def ids_to_sentence(ids, vocab_ids):
    return list(map(lambda id: vocab_ids[str(id)], ids))

# print(sentence_to_words(' Hôm nay là thứ hai, tháng mười một'))
# print(sentence_to_words(" I'm a student"))
# viet_vocab, viet_ids, num_viet_words = make_vocab('nmt_data/vie-eng-iwslt/train.vi')
# sentence_ids = sentence_to_ids(' Hôm nay là thứ hai, tháng mười một', viet_vocab)
# sentence = ids_to_sentence(sentence_ids, viet_ids)
# print(sentence_ids)
# print(sentence)


