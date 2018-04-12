from numpy import *


# Prepare Data: construct words vector from the text
def loadDataSet():
    # documents set where the words have been split
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # class label set
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union set
    return vocabSet


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[list(vocabList).index(word)] = 1
        else:
            print("The word: %s is not in my Vocabulary!" %word)
    return returnVec


# Train Algorithm: compute the propability from the words vector
def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])  # number of words in my vocabulary
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # propability of abusive docs
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = p1Num / p1Denom
    p0Vec = p0Num / p0Denom
    return p1Vec, p0Vec, pAbusive


# Test algorithm
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = 1
    p0 = 1
    # p(ci,W) = p(W|ci) * p(ci) = p(w0|ci) * p(w1|ci) * ... * p(wn|ci) * p(ci)
    pW_c1 = vec2Classify * p1Vec
    for i in pW_c1:
        p1 *= i
    p1 *= pClass1

    pW_c0 = vec2Classify * p0Vec
    for i in pW_c0:
        p0 *= i
    p0 *= 1.0 - pClass1

    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
        p0V, p1V, pAb = trainNB(trainMat, listClasses)

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

	testEntry = ['stupid', 'garbage']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

testingNB()