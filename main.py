from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import gensim
import numpy as np
from scipy import spatial
import time
from Calculate import avg_feature_vector
import pickle


def open_excel(path, model, set_, line_name):
    f = pd.read_excel(path, sheet_name=1, usecols=line_name)
    data = {}
    for line in range(len(f)):
        this_str = format(f.loc[line].values[0])
        vector = avg_feature_vector(sentence=this_str, model=model, num_features=100, index2word_set=set_)
        if np.any(vector):
            data[this_str] = vector
    return data


# 加载txt词向量文件, 并将其保存为二进制文件 https://blog.csdn.net/orangerfun/article/details/120253144
def load_word_vec(vec_path):
    wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=False)
    # 如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(vec_path.replace(".txt", ".bin"))


if __name__ == '__main__':
    t0 = time.time()
    embed_path = "tencent-ailab-embedding-zh-d100-v0.2.0-s.bin"
    wv_from_text = KeyedVectors.load(embed_path, mmap='r')
    index2word_set = set(wv_from_text.index_to_key)
    # source = open_excel("字段关联关系.xlsx", wv_from_text, index2word_set, 'D')
    with open('source.pkl', 'rb') as f:
        source = pickle.load(f)
    # query, k = input().split()
    # k = int(k)
    k = 5
    # test_vector = avg_feature_vector(query, wv_from_text, 100, index2word_set)
    test_vector = avg_feature_vector("姓名", wv_from_text, 100, index2word_set)
    my_dict = {}
    for i, (key, value) in enumerate(source.items()):
        sim = 1 - spatial.distance.cosine(test_vector, value)
        my_dict[key] = sim
    test_data_2 = sorted(my_dict.items(), key=lambda x: x[1], reverse=True)
    print(test_data_2[0:k])
    print(time.time() - t0)





    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # training()