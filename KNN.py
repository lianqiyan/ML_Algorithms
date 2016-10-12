import numpy as np
import operator
import matplotlib as plt

T_data = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
T_labels = ['A', 'A', 'B', 'B']

c_data = np.array([0, 0.2])
mdiff = np.tile(c_data, (T_data.shape[0], T_data.shape[1]-1)) - T_data
dist = (pow((pow(mdiff, 2).sum(axis=1)), 0.5))
or_index = np.argsort(dist)
num_lab = {}
for i in range(3):
    vlab = T_labels[or_index[i]]
    num_lab[vlab] = num_lab.get(vlab, 0) + 1
s_num = sorted(num_lab.items(), key=operator.itemgetter(1), reverse=True)
print(s_num[0])
