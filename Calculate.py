import jieba
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from scipy import spatial


def avg_feature_vector(sentence, model, num_features, index2word_set):
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


if __name__ == "__main__":
    """
       # tencent 预训练的词向量文件路径
       vec_path = "tencent-ailab-embedding-zh-d100-v0.2.0-s.txt"
       # 加载词向量文件
       wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=False)
       # 如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
       wv_from_text.init_sims(replace=True)
       wv_from_text.save(vec_path.replace(".txt", ".bin"))
       """
    embed_path = "tencent-ailab-embedding-zh-d200-v0.2.0-s.bin"
    wv_from_text = KeyedVectors.load(embed_path, mmap='r')
    index2word_set = set(wv_from_text.index_to_key)
    print(avg_feature_vector('侨居国', model=wv_from_text, num_features=100, index2word_set=index2word_set))
    s1_afv = avg_feature_vector('侨居国', model=wv_from_text, num_features=100, index2word_set=index2word_set)
    s2_afv = avg_feature_vector('收养人姓名', model=wv_from_text, num_features=100, index2word_set=index2word_set)
    sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
    print(sim)

