import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from Calculate import avg_feature_vector
import pandas as pd
from scipy import spatial
import Accuracy
import pickle


def open_excel(path, model, set_, line_name):
    f = pd.read_excel(path, sheet_name=1, usecols=line_name)
    data = {}
    for line in range(len(f)):
        this_str = format(f.loc[line].values[0])
        vector = avg_feature_vector(sentence=this_str, model=model, num_features=200, index2word_set=set_)
        if np.any(vector):
            data[this_str] = vector
    return data


if __name__ == "__main__":
    embed_path = "tencent-ailab-embedding-zh-d200-v0.2.0-s.bin"
    wv_from_text = KeyedVectors.load(embed_path, mmap='r')
    index2word_set = set(wv_from_text.index_to_key)
    source = open_excel("字段关联关系.xlsx", wv_from_text, index2word_set, 'D')
    with open('source.pkl', 'wb') as f:
        pickle.dump(source, f)
    test_vector = avg_feature_vector("学历", wv_from_text, 200, index2word_set)
    print(test_vector)
    my_dict = {}
    for i, (key, value) in enumerate(source.items()):
        sim = 1 - spatial.distance.cosine(test_vector, value)
        my_dict[key] = sim
    test_data_2 = sorted(my_dict.items(), key=lambda x: x[1], reverse=True)
    # Accuracy.accuracy(wv_from_text, index2word_set)
    print(test_data_2[0:10])
