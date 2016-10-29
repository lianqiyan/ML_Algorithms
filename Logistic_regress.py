import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def gradascent(data, classlabels):
    datamat = np.mat(data)
    labelmat = np.mat(classlabels).transpose()
    m, n = np.shape(datamat)
    alpha = 0.001
    maxcycle = 500
    weight = np.ones((n, 1))
    for i in range(maxcycle):
        h = sigmoid(datamat * weight)
        error = (labelmat - h)
        weight = weight + alpha * datamat.transpose() * error
    return weight


def stocgradascent(data, classlabels, num=150):
    data = np.mat(data)
    classlabels = np.mat(classlabels).transpose()
    m, n = np.shape(data)
    weights = np.ones((n, 1))
    dataindex = list(range(m))
    for j in range(num):
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randindex = int(np.random.uniform(0, len(dataindex)))
            h = sigmoid(np.sum(data[randindex] * weights))
            error = classlabels[randindex] - h
            weights = weights + data[randindex].transpose() * alpha * error
            # print(dataindex)
            # del(dataindex[randindex])
    return weights


def loaddata():
    datamat = []; labelmat = []
    f = open('testset.txt')
    for line in f.readlines():
        sline = line.strip().split()
        datamat.append([1.0, float(sline[0]), float(sline[1])])
        labelmat.append(int(sline[2]))
    return datamat, labelmat


def plotbestfit(weight):
    w = weight.getA()
    datamat, labelmat = loaddata()
    data = np.array(datamat)
    n = data.shape[0]
    xc1 = []; xc2 = []
    yc1 = []; yc2 = []
    for i in range(n):
        if int(labelmat[i]) == 1:
            xc1.append(data[i, 1]); yc1.append(data[i, 2])
        else:
            xc2.append(data[i, 1]); yc2.append(data[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xc1, yc1, s=30, c='red', marker='s')
    ax.scatter(xc2, yc2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-w[0] - w[1]*x)/w[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

data, labels = loaddata()
# wei = gradascent(data, labels)
wei = stocgradascent(data, np.array(labels), 500)
plotbestfit(wei)
