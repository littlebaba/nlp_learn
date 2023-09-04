import os


def load_data(path, flag='train'):
    """
    加载IMDB数据集
    路径aclImdb/train/neg|pos
    """
    labels = ['neg', 'pos']
    for label in labels:
        files = os.listdir(os.path.join(path, flag, label))  # 列出指定目录下的文件
        for file in files:
            with open(os.path.join(path, flag, label, file), 'r', encoding='utf-8') as rf:
                rf.read()
                a = 1

load_data(r"E:\data\aclImdb_v1\aclImdb")