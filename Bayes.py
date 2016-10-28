import numpy as np


def unique_word(data):
    words = set([])
    for word in data:
        words = words | set(word)
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


wList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmatian', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
classVec = [0, 1, 0, 1, 0, 1]
wordset = unique_word(wList)
print(wordset)
Mat = []
for i in wList:
    Mat.append(word2vector(wordset, i))
# print(Mat)
p0v, p1v, pAb = trainNBC(Mat, classVec)
print(p0v, '\n', p1v, '\n', pAb)
testword = [['stupid', 'garbage'], ['love', 'stop', 'dalmatian']]
t1 = word2vector(wordset, testword[0])
t2 = word2vector(wordset, testword[1])
print('t1 was classified as:  ', classifyNB(t1, p0v, p1v, pAb))
print('t2 was classified as:  ', classifyNB(t2, p0v, p1v, pAb))
