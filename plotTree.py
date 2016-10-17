import matplotlib.pyplot as plt


def plottext(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] + cntrPt[0])/2.0 + cntrPt[0]
    yMid  = (parentPt[1] + cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotree(tree, parentPt, nodetxt):
    numleafs = getnumleaf(tree)
    treedepth = getdepth(tree)
    firststr = list(tree.keys())[0]
    cntrPt = (plotree.xOff + (1.0 + float(numleafs))/2.0/plotree.totalW, plotree.yOff)
    plottext(cntrPt, parentPt, nodetxt)
    plotNode(firststr, cntrPt, parentPt, decision)
    secondDict = tree[firststr]
    plotree.yOff = plotree.yOff - 1.0/plotree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotree(secondDict[key], cntrPt, str(key))
        else:
            plotree.Off = plotree.xOff + 1.0/plotree.totalW
            plotNode(secondDict[key], (plotree.xOff, plotree.yOff), cntrPt, leafNode)
            plottext((plotree.xOff, plotree.yOff), cntrPt, str(key))
    plotree.yOff = plotree.yOff + 1.0/plotree.totalD


def plotNode(nodetext, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodetext, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotree.totalW = float(getnumleaf(mytree))
    plotree.totalD = float(getdepth(mytree))
    plotree.xOff = -0.5/plotree.totalW; plotree.yOff = 1.0
    plotree(mytree, (0.5, 1.0), '')
    # createPlot.ax1 = plt.subplot(111, frameon=False)
    # plotNode(U'decision node', (0.5, 0.1), (0.1, 0.5), decision)
    # plotNode(U'leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def getnumleaf(tree):
    num = 0
    firststr = list(tree.keys())[0]
    secondDict = tree[firststr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            num += getnumleaf(secondDict[key])
        else:
            num += 1
    return num


def getdepth(tree):
    depth = 0
    firststr = list(tree.keys())[0]
    secondDict = tree[firststr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            tdepth = 1 + getdepth(secondDict[key])
        else:
            tdepth = 1
        if tdepth > depth: depth = tdepth
    return depth


decision = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
mytree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
createPlot()
print(getnumleaf(mytree))
print(getdepth(mytree))
