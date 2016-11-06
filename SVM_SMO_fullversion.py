import numpy as np


class optStruct:
    def __init__(self, datain, classlablel, C, toler):
        self.labelmat = classlabels
        self.X = datain
        self.C = C
        self.tol = toler
        self.m = np.shape(datain)[0]
        self.alphas = np.mat((np.zeros(self.m, 1)))
        self.b = 0
        self.ecache = np.mat(np.zeros(self.m, 2))


def calcek(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelmat)).T * (oS.X*oS.X[k, :] + oS.b)
    Ek = fXk - float(oS.labelmat[k])
    return Ek


def selectJ(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCahe[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.ecache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
             if k == i: continue
             Ek = calcek(oS, k)
             deltaE = abs(Ei - Ek)
             if (deltaE > maxDeltaE):
                 maxK = k; maxDeltaE = deltaE; Ej = Ek
    return maxK, Ej


def updateEk(oS, k):
    Ek = calcek(oS, k)
    oS.ecache[k] = [1, Ek]


def clipAlpha(aj, h, l):
    if aj > h:
        aj = h
    if l > aj:
        aj = l
    return aj


def inner(i, oS):
    Ei = calcek(oS, i)
    if ((oS.labesmat[i] < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelmat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaiOLD = oS.alphas[i].copy(); alphajOLD = oS.alphas[j].copy()
        if oS.labelmat[i] != oS.alphas[i]:
            L = max(0, oS.alphas[j] - oS.alphas[i]):
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L = H")
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[i].T - oS.X[i,:]*oS.x[i, :].T - oS.X[j, :] * oS.X[j,:].T
        if eta >= 0: print("eta >= 0"); return 0
        oS.alphas[j] -= oS.alphals[j] * (Ei -Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]- alphaiOLD)*oS.X[i, :]*oS.X[i, :].T - \
        oS.labelsmat[j]*(oS.X[j] - alphajOLD)*oS.X[i, :]*oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]- alphaiOLD)*oS.X[i, :]*oS.X[j, :].T - \
        oS.labelsmat[j]*(oS.X[j] - alphajOLD)*oS.X[j, :]*oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else: return 0


def smoPK(dataMatIn, classLabels, C, toler, maxIter):  # full Platt SMO
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True;
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d", iter, i, alphaPairsChanged)
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d", iter, i, alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas
