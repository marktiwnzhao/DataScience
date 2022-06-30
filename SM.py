import pandas as pd
import xlrd
import difflib


def get_equal_rate_1(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).quick_ratio()

def findTopk_SM(query_str, topk):
    k = topk
    f = pd.read_excel("字段关联关系.xlsx", sheet_name=0, usecols='B')
    size = len(f)
    # query是待匹配的字符串
    query = query_str
    similarity_dict = {}
    for line in range(size):
        b_str = format(f.loc[line].values[0])
        # similarity_list.append(get_equal_rate_1(query, b_str))
        similarity_dict[b_str] = get_equal_rate_1(query, b_str)
    res = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)
    print(res[0:k])

def findTop1_SM(query_str):
    f = pd.read_excel("字段关联关系.xlsx", sheet_name=0, usecols='B')
    size = len(f)
    # query是待匹配的字符串
    query = query_str
    similarity_dict = {}
    for line in range(size):
        b_str = format(f.loc[line].values[0])
        similarity_dict[b_str] = get_equal_rate_1(query, b_str)
    res = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)
    return res[0][0]


if __name__ == "__main__":
    findTopk_SM("所在行政区", 5)





