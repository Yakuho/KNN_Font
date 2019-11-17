# -*- coding: utf-8 -*-
# Author: Yakuho
# Date  : 2019/11/15
import numpy as np
import operator


def classify_knn(input_x: np.ndarray or list, dataSet: np.ndarray or list, labels: np.ndarray or list, k:int)->list:
    '''
    :param input_x:     data for predicting
    :param dataSet:     dataset for training
    :param labels:   The labels of dataset
    :param k:   The range of a point of input_x
    :return list:  A list of probability which have minimum distance

    Either input_x or dataSet, they're should to normalize at first
    '''
    dataSet = np.array(dataSet)
    labels = np.array(labels).flatten()  # flattening (扁平化)
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(input_x, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  # Turning the sum of every rows to 1*dataSetSize matrix
    distances = sqDistances**0.5
    # Return a sorted(fr min to huge) index list of the value of 1*dataSetSize matrix
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):  # Place k numbers of labels/type to classCount keys, and add the values of classCount[label[k]]
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # sorted by values(the labels times)
    return sortedClassCount[0][0]

def normalize_dataset_01(dataset: list or np.ndarray, mindata: list or np.ndarray=[], maxdata: list or np.ndarray=[]):
    '''
    :param dataset: dataset for normalizing
    :param mindata: The matrix of minimum values
    :param maxdata: The matrix of maximum values
    :return:
    '''
    dataset = np.array(dataset)
    # When sample unstable probably, means that sample over max or less than minimum, U need to set values by yourself
    if not (mindata and maxdata):
        mindata = dataset.min(0)    # 0 for figuring the minimum of every columns (the same as max)
        maxdata = dataset.max(0)    # 1 for figuring the minimum of every rows (the same as max)
    else:
        maxdata = np.array(maxdata)
        mindata = np.array(mindata)
    rangedata = maxdata - mindata
    # the same as ↓↓↓(dataset - mindata) / np.tile(rangedata, (dataset.shape[0], 1)↓↓↓
    # due to python auto turn X1 = 1*n matrix to n*n matrix when X1 divide by X2(n*n)
    dataset = (dataset - mindata) / rangedata  # )
    return dataset, mindata, rangedata

def normalize_data_01(data: list or np.ndarray, mindata: list or np.ndarray, rangedata: list or np.ndarray):
    '''
    :param data: dataset for normalizing
    :param mindata: The matrix of minimum values
    :param rangedata: The matrix of max subtraction min of every columns
    :return:
    '''
    data = np.array(data)
    return (data - mindata)/rangedata

def normalize_dataset_z_score(dataset: list or np.ndarray):
    '''
    :param dataset: dataset for normalizing
    :return:
    '''
    dataset = np.array(dataset)
    meansMat = dataset.sum(axis=0)/dataset.shape[0] # get means
    diffMat = dataset - meansMat
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=0)/dataset.shape[0]
    standardDeviation = sqDistances ** 0.5  # get standard deviation
    dataset = (dataset - meansMat) / standardDeviation
    return meansMat, standardDeviation, dataset

def normalize_data_z_score(dataset: list or np.ndarray, meansMat: list or np.ndarray, standardDeviation: list or np.ndarray):
    '''
    :param dataset: dataset for normalizing
    :param meansMat: meansMat of training dataset
    :param standardDeviation: standardDeviation of training dataset
    :return:
    '''
    dataset = np.array(dataset)
    return (dataset - meansMat) / standardDeviation

def normalize_data_z_score_arctan(dataset: list or np.ndarray):
    '''
    :param dataset: dataset for normalizing
    :return:
    '''
    dataset = np.array(dataset)
    dataSet = np.arctan(dataset)*(2/np.pi)
    return dataSet
