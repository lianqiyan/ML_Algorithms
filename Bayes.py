import numpy as np


def unique_word(data):
    wordset = set([])
    for word in data:
        wordset = wordset | set(word)
    return list(wordset)


def word2vector(wordlist, inputset):
    vector = [0] * len(wordlist)
    for word in wordlist:
        if word in inputset:
            vector[wordlist.index(word)] = 1
        else:
            print("The word is not in my vovabulary!", word)
    return vector


def trainNBC(Matrix, Category):
    Num = len(Matrix)
    Numw = len(Matrix[0])
    PA = sum(Category)/float(Num)
    P0 = np.zeros(Numw); P1 = np.zeros(Numw)
    P0D = 0.0; P1D =0.0
    for i in range(Num):
        if Category[i] == 1:
            P1 += Matrix[i]
            P1D += sum(Matrix[i])
        else:
            P0 += Matrix[i]
            P0D += sum(Matrix[i])
    P1V = P1/P1D
    P0V = P0/P0D
    return P0V, P1V, PA


wList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
classVec = [0, 1, 0, 1, 0, 1]
wordset = unique_word(wList)
Mat = []
for i in wList:
    Mat.append(word2vector(wordset, i))
p0V, p1V, pAb = trainNBC(Mat, classVec)
print(p0V, p1V, pAb)
