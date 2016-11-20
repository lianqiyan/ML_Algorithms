import numpy as np
import matplotlib.pyplot as plt


def loadata(name):
    a = []
    f = open(name)
    for line in f.readlines():
        line = line.strip().split()
        temp = [float(i) for i in line]
        a.append(temp)
    return np.array(a)


def splitdata(mat):
    s_list = []
    all_value = np.unique(mat[:, -1])
    for value in all_value:
        temp = mat[mat[:, 2]==value, 0:-1]
        s_list.append(temp)
    return s_list


def distance(a, b):
    return np.sqrt(sum(np.power(a-b, 2)))


def extract_data(data):
    full_data = np.zeros(np.shape(data))
    full_data[:, 0:-1] = data[:, 0:-1]
    full_data[:, -1] = None
    return full_data


def generate_center(data, k):
    n = np.shape(data)[1]
    center = np.zeros((k, n))
    for i in range(n):
        minI = min(data[:, i])
        rangeI = max(data[:, i]) - minI
        center[:, i] = minI + rangeI * np.random.rand(k)
    return center


def k_means(data, k):
    # index = np.random.randint(0, np.shape(data)[0], k)
    # print(index)
    # center = data[index, 0:-1]
    center = generate_center(data[:, 0:-1], k)
    changed = True
    while changed==True:
        for i in range(0, np.shape(data)[0]):
            dis = distance(center, data[i, 0:-1])
            index = np.argsort(dis)
            data[i, -1] = index[0]
        new_center = np.zeros(np.shape(center))
        for i in range(0, np.shape(center)[0]):
            temp = data[data[:, -1]==i, 0:-1]
            new_center[i, :] = np.mean(temp, 0)
        print(center)
        print(new_center)
        print(abs(new_center - center).sum())
        if abs(new_center - center).sum() < 0.2:
            changed = False
        else:
            changed = True
            center = new_center
    return data, center


dat = loadata('testSet.txt')
dat = extract_data(dat)
c_data, c = k_means(dat, 2)
all = splitdata(c_data)
a1 = all[0]; a2 = all[1]
plt.scatter(a1[:, 0], a1[:, 1], marker="o", color='green')
plt.scatter(a2[:, 0], a2[:, 1], marker="o", color='red')
plt.scatter(c[:, 0], c[:, 1], marker="o", color='blue')
plt.show()
