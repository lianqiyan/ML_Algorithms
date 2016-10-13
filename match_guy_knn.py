import numpy as np
import matplotlib.pyplot as plt
import operator


def file2mat():
    f = open('datingTestSet2.txt')
    a_lines = f.readlines()
    num_lines = len(a_lines)
    mat = np.zeros((num_lines, 3))
    index = 0
    labels = []
    for line in a_lines:
        line = line.strip()
        lis_line = line.split('\t')
        mat[index, :] = lis_line[0:3]
        index += 1
        labels.append(int(lis_line[-1]))
    return mat, labels


def normdat(data):
    c_max = data.max(0)
    c_min = data.min(0)
    c_range = c_max - c_min
    nor_dat = data - np.tile(c_min, (data.shape[0], 1))
    nor_dat = nor_dat/np.tile(c_range, (data.shape[0], 1))
    return nor_dat, c_range, c_min


def knn_classify(T_data, T_labels, k, c_data):
    mdiff = np.tile(c_data, (T_data.shape[0], 1)) - T_data
    dist = (pow((pow(mdiff, 2).sum(axis=1)), 0.5))
    or_index = np.argsort(dist)
    num_lab = {}
    for i in range(k):
        vlab = T_labels[or_index[i]]
        num_lab[vlab] = num_lab.get(vlab, 0) + 1
    s_num = sorted(num_lab.items(), key=operator.itemgetter(1), reverse=True)
    return s_num[0][0]


def test_classify():
    count = 0
    data, labels = file2mat()
    data, a, b = normdat(data)
    for row in range(len(labels)):
        c_lab = knn_classify(data, labels, 5, data[row, :])
        if c_lab is labels[row]:
            continue
        else:
            count += 1
    rate = count / len(labels)
    return rate


def classify_person():
    percetagetat = float(input("percentage of time spent on playing games?"))
    icecream = float(input("How much does the guy eat ice cream?"))
    fmiles = float(input("miles fly per year?"))
    data = np.array([percetagetat, icecream, fmiles])
    print(data)
    tdata, labels = file2mat()
    tdata, t_range, t_min = normdat(tdata)
    data = (data - t_min)/t_range
    c_lab = knn_classify(tdata, labels, 5, data)
    print(c_lab)
    if c_lab is '1':
        print("YOU WON'T LIKE THE GUY AT ALL")
    elif c_lab is '2':
        print("YOU MAY LIKE THE GUY A BIT")
    else:
        print("YOU MAY LIKE THE GUY VERY MUCH")


print("The corret rate is", test_classify())
classify_person()

a, b = file2mat()
print(normdat(a)[1])
plt.scatter(a[:, 1], a[:, 2], 15*np.array(b), 15*np.array(b))
plt.show()
plt.scatter(a[:, 0], a[:, 1], 15*np.array(b), 15*np.array(b))
plt.show()

