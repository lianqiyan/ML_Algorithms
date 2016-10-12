import numpy as np
import operator
import matplotlib as plt

p_data = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
p_labels = ['A', 'A', 'B', 'B']


def knn_classify(T_data, T_labels, k, c_data):
    c_labels = []
    for i in range(c_data.shape[0]):
        temp = c_data[i][:]
        mdiff = np.tile(temp, (T_data.shape[0], T_data.shape[1] - 1)) - T_data
        dist = (pow((pow(mdiff, 2).sum(axis=1)), 0.5))
        or_index = np.argsort(dist)
        num_lab = {}
        for i in range(k):
            vlab = T_labels[or_index[i]]
            num_lab[vlab] = num_lab.get(vlab, 0) + 1
        s_num = sorted(num_lab.items(), key=operator.itemgetter(1), reverse=True)
        c_labels.append(s_num[0][0])
    return c_labels

t_data = np.array([[1.2, 0.8], [-0.1, 0.3]])
print(knn_classify(p_data, p_labels, 3, t_data))
