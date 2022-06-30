import pickle
import jieba
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


class RenewDict:
    def __init__(self, embed_path, filename):
        self.embed_path = embed_path
        self.filename = filename

    def avg_feature_vector(self, sentence, model, num_features, index2word_set):
        words = jieba.lcut(sentence)
        feature_vec = np.zeros((num_features,), dtype='float32')
        n_words = 0
        for word in words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word])
        if n_words > 0:
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    def open_excel(self, path, model, set_, line_name):
        f = pd.read_excel(path, sheet_name=1, usecols=line_name)
        data = {}
        for line in range(len(f)):
            this_str = format(f.loc[line].values[0])
            vector = self.avg_feature_vector(sentence=this_str, model=model, num_features=200, index2word_set=set_)
            if np.any(vector):
                data[this_str] = vector
        return data

    def renew(self):
        wv_from_text = KeyedVectors.load(self.embed_path, mmap='r')
        index2word_set = set(wv_from_text.index_to_key)
        source = self.open_excel("字段关联关系.xlsx", wv_from_text, index2word_set, 'D')
        with open('source.pkl', 'wb') as f:
            pickle.dump(source, f)
        print("Renew done!")
