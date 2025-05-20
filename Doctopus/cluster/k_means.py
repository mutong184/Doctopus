
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from cluster.text2emb import get_dataset_embdding
import numpy as np

from collections import defaultdict
import itertools


import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import random

def get_kmeans(file_dict, n_clusters, random_state=2024):
    # 参数验证
    if not file_dict or not isinstance(file_dict, dict):
        raise ValueError("file_dict must be a non-empty dictionary.")
    if n_clusters <= 0 or n_clusters > len(file_dict):
        raise ValueError("n_clusters must be a positive integer less than or equal to the number of files.")

    # 向量化
    file2embeding = file_dict
    filenames = list(file2embeding.keys())
    vectors = np.stack([t.numpy() for t in file2embeding.values()])  # 转换为 (n_samples, 4096) 的矩阵

    # K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(vectors)

    # 分组
    label_to_files = defaultdict(list)
    for filename, label in zip(filenames, labels):
        label_to_files[label].append(filename)

    return dict(sorted(label_to_files.items()))  # 按 label 排序返回


def sample_by_percentage(file_dict, percentage):

    if not (0 < percentage <= 1):
        raise ValueError("Percentage should be a value between 0 and 1.")
    
    sampled_dict = {}
    
    for category, file_list in file_dict.items():
        num_to_sample = max(1, int(len(file_list) * percentage))  # 至少取一个文件
        sampled_files = random.sample(file_list, num_to_sample)
        sampled_dict[category] = sampled_files
    
    return sampled_dict


