import math
import operator
import matplotlib.pyplot as plt


def cal_entropy(data):
    num = len(data)
    labcount = {}
    for vec in data:
        c_label = vec[-1]
        if c_label not in labcount.keys():
            labcount[c_label] = 0
        labcount[c_label] += 1
    entropy = 0
    for key in labcount:
        prob = float(labcount[key])/num
        entropy -= prob * math.log(prob, 2)
    return entropy


def spilt_data(data, axis, value):
    rdata = []
    for vec in data:
        if vec[axis] == value:
            rfv = vec[:axis]
            rfv.extend(vec[axis+1:])
            rdata.append(rfv)
    return rdata


def choosebestf(data):
    num = len(data[0]) - 1
    entropy = cal_entropy(data)
    bestinfogain = 0.0
    bestF = -1
    for i in range(num):
        flist = [example[i] for example in data]
        uniqueval = set(flist)
        newentropy = 0.0
        for value in uniqueval:
            subdataset = spilt_data(data, i, value) # generate new sub dataset
            prob = len(subdataset)/float(len(data))
            newentropy += prob * cal_entropy(subdataset)
        infogain = entropy - newentropy  # information gain
        if infogain > bestinfogain:  # get the best feature by compare
            bestinfogain = infogain
            bestF = i
    return bestF


def majorityCnt(classlist):
    classcount = {}
    for v in classlist:
        if v not in classcount.keys(): classcount[v] = 0
        classcount[v] += 1
    sortedClassCount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createtree(data, label):
    classlist = [example[-1] for example in data]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]   # if the rest of sample are in the same class, then stop classify
    if len(data[0]) == 1:   # if end iterating all feature, then return the most one
        return majorityCnt(classlist)
    bfeature = choosebestf(data)
    blabel = label[bfeature]
    print('BEST:', bfeature, blabel)
    mytree = {blabel: {}}
    del(label[bfeature])
    fvalue = [example[bfeature] for example in dataset]
    uniquevals = set(fvalue)
    for value in uniquevals:
        sublabels = labels[:]
        mytree[blabel][value] = createtree(spilt_data(data, bfeature, value), sublabels)
        print(spilt_data(data, bfeature, value))
    return mytree


def classify(inputTree, featlabels, testvec):
    print(featlabels)
    firststr = list(inputTree.keys())
    secondDict = inputTree[firststr[0]]
    featIndex = 0 # get index
    for key in secondDict.keys():
        if testvec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classlabel = classify(secondDict[key], featlabels, testvec)
            else:  classlabel = secondDict[key]
        featIndex += 1
    return classlabel

dataset = [[1, 1, 'yes'],
           [1, 1, 'yes'],
           [1, 0, 'no'],
           [0, 1, 'no'],
           [0, 1, 'no']]
labels = ['no surfacing', 'flippers']

print(cal_entropy(dataset))
print(cal_entropy(spilt_data(dataset, 0, 1)))
print(spilt_data(dataset, 0, 1))
print(choosebestf(dataset))
# print(createtree(dataset, labels))
tree = createtree(dataset, labels)
print(tree)
print(classify(tree, labels, [1, 1]))
