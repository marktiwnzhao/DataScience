import numpy as np
from gensim.models import KeyedVectors

from Calculate import avg_feature_vector
import pandas as pd
from scipy import spatial
from Test import open_excel
import csv
from SM import findTop1_SM


def accuracy(model, set_):
    filename = open("错误数据元.csv", 'w', encoding="utf-8")
    csv_writer = csv.writer(filename)
    csv_writer.writerow(["行数", "数据元", "标准数据元", "错误结果"])
    source = open_excel("字段关联关系.xlsx", model, set_, 'D')
    f1 = pd.read_excel("字段关联关系.xlsx", sheet_name=1, usecols='C')
    f2 = pd.read_excel("字段关联关系.xlsx", sheet_name=1, usecols='D')
    cnt = 0
    for line in range(len(f1)):
        test_data = format(f1.loc[line].values[0])
        vector = avg_feature_vector(sentence=test_data, model=model, num_features=200, index2word_set=set_)
        if np.any(vector):
            my_dict = {}
            for i, (key, value) in enumerate(source.items()):
                sim = 1 - spatial.distance.cosine(vector, value)
                my_dict[key] = sim
            test_data_2 = sorted(my_dict.items(), key=lambda x: x[1], reverse=True)
            data_of_SM = findTop1_SM(test_data)
            if test_data_2[0][0] == format(f2.loc[line].values[0]) or data_of_SM == format(f2.loc[line].values[0]):
                cnt += 1
            else:
                csv_writer.writerow([line, test_data, format(f2.loc[line].values[0]), test_data_2[0][0]])
            print(cnt)
    filename.close()
    print("New Results")
    print(cnt / 5891)
    return
