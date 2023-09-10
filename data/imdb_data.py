import os
import re
import time
from concurrent.futures.thread import ThreadPoolExecutor

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


def process_sentence(flag):
    '''
    将单词组成的句子映射成单词编号向量，即句子编码
    '''
    word_list = np.load('npys/word_list.npy')
    word_list = word_list.tolist()
    data = load_data(path, flag)
    sentence_code = []

    def task(d):  # 耗时操作
        vec = d[0]
        tmp = []
        index = 0
        for v in vec:  # 查每个单词在word_list中的索引，若存在则加入的tmp中，若不存在则按索引399999赋值
            try:
                index = word_list.index(v)
            except ValueError:  # 没找到
                index = 399999
            finally:
                tmp.append(index)
        # 如果句子长度小于250则末尾补0，若大于250则大于部分去掉
        if len(tmp) < 250:
            for k in range(len(tmp), 250):
                tmp.append(0)
        else:
            tmp = tmp[:250]
        sentence_code.append(tmp)

    pool = ThreadPoolExecutor(max_workers=6)
    for d in data:
        pool.submit(task, d)
        a = 1

    while len(sentence_code) != len(data):
        time.sleep(1)

    # 转矩阵保存二进制进硬盘
    sentence_code = np.array(sentence_code)
    if flag == 'train':
        np.save("npys/sentence_code_1", sentence_code)
    else:
        np.save("npys/sentence_code_2", sentence_code)


def proccess_batch(batch_size):
    '''
    4万个句子分批，每一批句子的大小为100
    '''
    # 加载词表
    vocabulary_vectors = np.load('npys/vocabulary_vectors.npy')
    vocabulary_vectors = vocabulary_vectors.tolist()
    # 加载句子
    sentence_code_1 = np.load('npys/sentence_code_1.npy')
    sentence_code_1 = sentence_code_1.tolist()
    for i in range(25000):
        sentence_code_1[i] = [vocabulary_vectors[word_idx] for word_idx in sentence_code_1[i]]

    # 加载原始数据
    train_data = load_data(path)
    arr_train = []
    labels_train = []
    for i in range(1, int(len(train_data) / batch_size) + 1):
        arr_train.append(sentence_code_1[(i - 1) * batch_size:i * batch_size])  # 加批数据
        labels_train.append([train_data[k][1] for k in range((i - 1) * batch_size, i * batch_size)])  # 加批标签
    arr_train = np.array(arr_train)
    labels_train = np.array(labels_train)
    np.save('npys/arr_train', arr_train)
    np.save('npys/labels_train', labels_train)
    return arr_train, labels_train


proccess_batch(100)
