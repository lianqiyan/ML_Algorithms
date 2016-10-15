import math


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
    num = len(data[0])
    entropy = cal_entropy(data)
    bestinfogain = 0.0
    bestF = 0.0
    for i in range(num):
        flist = [example[i] for example in data]
        uniqueval = set(flist)
        newentropy = 0.0
        for value in uniqueval:
            subdataset = spilt_data(data, i, value)
            prob = len(subdataset)/float(len(data))
            newentropy += prob * cal_entropy(subdataset)
        infogain = entropy - newentropy
        if infogain > bestinfogain:
            bestinfogain = infogain
            bestF = i
    return bestF

dataset = [[1, 1, 'yes'],
           [1, 1, 'yes'],
           [1, 0, 'no'],
           [0, 1, 'no'],
           [0, 1, 'no']]
labels = ['no surface', 'flippers']

print(cal_entropy(dataset))
print(spilt_data(dataset, 0, 1))
print(choosebestf(dataset))
