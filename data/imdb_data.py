import os
import re

import numpy as np

root = r"e:\data\aclImdb_v1"
path = os.path.join(root, 'aclImdb')


def load_data(path, flag='train'):
    """
    加载IMDB数据集
    路径aclImdb/train/neg|pos
    """

    labels = ['neg', 'pos']
    data = []
    for label in labels:
        files = os.listdir(os.path.join(path, flag, label))  # 列出指定目录下的文件
        r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
        for file in files:
            with open(os.path.join(path, flag, label, file), 'r', encoding='utf-8') as rf:
                tmp = rf.read().replace('\n', '')  # 去换行
                tmp = re.sub(r, '', tmp)  # 去符号
                tmp = tmp.split(' ')  # 句子拆分成单词
                tmp = [tmp[i].lower() for i in range(len(tmp))]  # 单词转小写
                # 单词向量末尾追加label
                if label == 'pos':
                    data.append([tmp, 1])
                elif label == 'neg':
                    data.append([tmp, 0])
    return data


def load_cab_vector(root):
    '''
    40万个单词，每个单词用长度为50的向量表示
    '''
    word_list = []
    vocabulary_vectors = []
    data = open(os.path.join(root, 'glove.6B.50d.txt'), encoding='utf-8')
    for line in data.readlines():  # 40万个单词
        tmp = line.strip('\n').split(' ')
        name = tmp[0]
        word_list.append(name.lower())
        vector = tmp[1:]  # 长度为50的词向量
        # 字符串转浮点数
        vector = list(map(float, vector))
        vocabulary_vectors.append(vector)
    vocabulary_vectors = np.array(vocabulary_vectors)  # 二维
    word_list = np.array(word_list)  # 一维
    if not os.path.exists('npys'):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs('npys')
    np.save('npys/vocabulary_vectors', vocabulary_vectors)
    np.save('npys/word_list', word_list)
    return vocabulary_vectors, word_list


load_cab_vector()
