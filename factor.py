import numpy as np
import pandas as pd


def shannon_entropy(data, bins=5):
    # 把连续数值切成 bins 段
    categories = pd.cut(data, bins=bins, labels=False)
    probs = pd.Series(categories).value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))


def construct_templates(timeseries_data, m):  # 时间序列分割
    num_windows = len(timeseries_data) - m + 1
    return np.array([timeseries_data[x : x + m] for x in range(0, num_windows)])


def fuzzy_membership(dist, r, n=2):
    return np.exp(- (dist ** n) / r)


def get_matches(templates, r):
    return len(list(filter(lambda x: is_match(x[0], x[1], r), combinations(templates))))


def combinations(x):  # 生成上三角矩阵索引
    idx = np.stack(np.triu_indices(len(x), k=1), axis=-1)
    return x[idx]


def is_match(template_1, template_2, r):
    return np.all([abs(x - y) < r for (x, y) in zip(template_1, template_2)])  # 切比雪夫距离


# def is_match(template_1, template_2, r):
#     euclidean_distance = np.linalg.norm(template_1 - template_2)
#     return euclidean_distance < r  # 欧几里得距离


def sample_entropy(timeseries_data, window_size, r):  # r是0.1到0.25倍的数据标准差
    r *= float(np.std(timeseries_data))
    B = get_matches(construct_templates(timeseries_data, window_size), r)
    A = get_matches(construct_templates(timeseries_data, window_size + 1), r)

    if B == 0:
        return np.inf
    if A == 0:
        return -np.inf
    return -np.log(A / B)