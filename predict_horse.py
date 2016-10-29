import numpy as np


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def stocgradascent(data, classlabels, num=150):
    data = np.mat(data)
    classlabels = np.mat(classlabels).transpose()
    m, n = np.shape(data)
    weights = np.ones((n, 1))
    dataindex = list(range(m))
    for j in range(num):
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randindex = int(np.random.uniform(0.0, len(dataindex)))
            t = np.tile(0, data[randindex].shape)
            # print(t, data[randindex])
            if np.sum(t == data[randindex]) == 0:
                h = sigmoid(np.sum(data[randindex] * weights))
                error = classlabels[randindex] - h
                weights = weights + data[randindex].transpose() * alpha * error
            # else:
            #     print('Exist')
    return weights


def classify(x, weights):
    prob = sigmoid(sum(x * weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def colictest():
    ftest = open('horseColicTest.txt')
    ftrain = open('horseColicTraining.txt')
    trainset = []; trainlabel= []
    for line in ftrain.readlines():
        line = line.strip().split()
        s_line = []
        for i in range(21):
            s_line.append(float(line[i]))
        trainset.append(s_line)
        trainlabel.append(float(line[21]))
    t_weight = stocgradascent(np.array(trainset), trainlabel, 500)
    errorcount = 0; numtestvec = 0.0
    for line in ftest.readlines():
        numtestvec += 1.0
        currline = line.strip().split()
        s_line = []
        for i in range(21):
            s_line.append(float(currline[i]))
        if int(classify(np.array(s_line), t_weight)) != int(currline[21]):
            errorcount += 1
    errorate = (float(errorcount)/numtestvec)
    print('The error rate of this test is: ', errorate)
    return errorate


def multitest():
    num_test = 10; errorsum = 0.0
    for k in range(num_test):
        errorsum += colictest()
    print('After ' + str(num_test) + 'itrations the average error is :' + str(errorsum/float(num_test)))


multitest()
