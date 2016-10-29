import numpy as np
import re


def unique_word(data):
    words = set([])
    for wor in data:
        words = words | set(wor)
    return list(words)


def word2vector(wordlist, inputset):
    vector = [0] * len(wordlist)
    for word in wordlist:
        if word in inputset:
            vector[wordlist.index(word)] = 1
        # else:
        #     print("The word is not in my vocabulary!", word)
    return np.array(vector)


def trainNBC(Matrix, Category):
    Num = len(Matrix)
    Numw = len(Matrix[0])
    PA = sum(Category)/float(Num)  # possibility of bad words
    P0 = np.ones(Numw); P1 = np.ones(Numw)  # to avoid multiplying zeros so set p to one
    P0D = 2.0; P1D = 2.0
    for i in range(Num):
        if Category[i] == 1:
            P1 += Matrix[i]  # number of  every words which appear in good labels
            P1D += sum(Matrix[i])  # number of word which appear in good labels
        else:
            P0 += Matrix[i]
            P0D += sum(Matrix[i])
    P1V = np.log(P1/P1D)
    P0V = np.log(P0/P0D)  # use log to avoid multiplying a very small number
    return P0V, P1V, PA


def classifyNB(vector, p0vec, p1vec, pclass1):
    p1 = sum(vector * p1vec) + np.log(pclass1)
    p2 = sum(vector * p0vec) + np.log(1 - pclass1)
    if p1 > p2:
        return 1
    else:
        return 0


def textparse(string):
    s_str = re.split('\W', string)
    s_str = [x for x in s_str if len(x) > 2]
    return s_str


doclist = []; classlist = []; fulltext = []
for i in range(1, 26):
    word = textparse(open('E:/python code/MachineLearning/spam/'+str(i)+'.txt').read())
    doclist.append(word)  # mail list
    fulltext.extend(word)  # all words in a list
    classlist.append(1)
    word = textparse(open('E:/python code/MachineLearning/ham/'+str(i)+'.txt').read())
    doclist.append(word)
    fulltext.extend(word)
    classlist.append(0)
vocablist = unique_word(doclist)
trainset = list(range(50)); testset = []
for i in range(10):  # generate 10 test data
    randindex = int(np.random.uniform(0, len(trainset)))
    testset.append(trainset[randindex])
    del(trainset[randindex])
trainmat = []; trainclass = []
for docindex in trainset:
    trainmat.append(word2vector(vocablist, doclist[docindex]))
    trainclass.append(classlist[docindex])
p0v, p1v, pspam = trainNBC(np.array(trainmat), np.array(trainclass))
errorcount = 0
for docindex in testset:
    wvector = word2vector(vocablist, doclist[docindex])
    if classifyNB(wvector, p0v, p1v, pspam) != classlist[docindex]:
        errorcount += 1
print('The error rate is ', float(errorcount)/len(testset))
