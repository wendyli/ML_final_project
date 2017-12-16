#!/usr/bin/python

import import_data
import math
import numpy
import collections
import sys
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc


model_to_use = 'logistic'
#model_to_use = 'random'

def generateFileNames(dir, year, months, days):
    filenames = []
    yearStr = str(year)
    for i in range(0, len(months)):
        monthStr = '0' + str(months[i]) if months[i] < 10 else str(months[i])
        for j in range(1, days[i]+1):
            dayStr = '0' + str(j) if j < 10 else str(j)
            file_data_name = dir + "{0}-{1}-{2}_smoothed_10.csv".format(yearStr, monthStr, dayStr)
            filenames.append(file_data_name)
    
    return filenames


def generateFileNamesNoTest(dir, year, months, days):
    filenames = []
    yearStr = str(year)
    for i in range(0, len(months)):
        monthStr = '0' + str(months[i]) if months[i] < 10 else str(months[i])
        for j in range(1, days[i]+1):
            dayStr = '0' + str(j) if j < 10 else str(j)
            file_data_name = dir + "{0}-{1}-{2}_smoothed_0.csv".format(yearStr, monthStr, dayStr)
            filenames.append(file_data_name)
    
    return filenames

# Parameters:
#   data: (featureNames, map of serial number to disk data point)
#   features: set of features to train on
#
# Returns the train regression model
def trainLogisticRegression(dataPointsX, dataPointsY):
    #reg = linear_model.LogisticRegression(class_weight='balanced')
    reg = linear_model.LogisticRegression()
    reg.fit(dataPointsX, dataPointsY)
    return reg


# Parameters:
#   data: (featureNames, map of serial number to disk data point)
#   features: set of features to train on
#
# Returns the trained random forest model
def trainRandomForest(dataPointsX, dataPointsY):
    forest = RandomForestClassifier(max_depth=4, random_state=0)
    forest.fit(dataPointsX, dataPointsY)
    return forest


# Define feature processing functions
def process_smart_raw_default(smart_raw_string):
    if smart_raw_string == '':
        return 0
    
    return int(smart_raw_string)


def critical_value_processed(value, weight):
    return 1.0 - 1.0 / (weight * value + 1.0)


# Reallocated Sectors Count (critical value)
def process_smart_5_raw(smart_5_raw_string):
    if smart_5_raw_string == '':
        return 0
    
    return critical_value_processed(int(smart_5_raw_string), 0.5)

# Power On Hours
def process_smart_9_raw(smart_9_raw_string):
    if smart_9_raw_string == '':
        return 0
    
    return int(smart_9_raw_string) / 240

# Spin Retry Count
def process_smart_10_raw(smart_10_raw_string):
    if smart_10_raw_string == '':
        return 0
    
    return critical_value_processed(int(smart_10_raw_string), 0.05)


# Reported Uncorrectable Errors
def process_smart_187_raw(smart_187_raw_string):
    if smart_187_raw_string == '':
        return 0
    
    return critical_value_processed(int(smart_187_raw_string), 0.5)


# Command Timeout
def process_smart_188_raw(smart_188_raw_string):
    if smart_188_raw_string == '':
        return 0
    
    return critical_value_processed(int(smart_188_raw_string), 2.0)


# (Offline) Uncorrectable Sector Count
def process_smart_198_raw(smart_198_raw_string):
    if smart_198_raw_string == '':
        return 0
    
    return critical_value_processed(int(smart_198_raw_string), 1.0)


def print_stats_helper(prefix_msg, numerator, denominator):
    if denominator > 0:
        prefix_msg += str(numerator / denominator)
    else:
        prefix_msg += "undefined"
    print prefix_msg


def generate_roc(actual, predicted):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predicted)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('ROC Curve (Logistic Regression)')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# Assumes the arguments are floats
def print_stats(TP, TN, FP, FN):
    print "True positive: " + str(TP)
    print "True negative: " + str(TN)
    print "False positive: " + str(FP)
    print "False negative: " + str(FN)

    print_stats_helper("True positive rate (TP / (TP + FN)): ", TP, TP + FN)
    print_stats_helper("True negative rate (TN / (TN + FN)): ", TN, TN + FN)
    print_stats_helper("False positive rate (FP / (FP + TN)): ", FP, FP + TN)
    print_stats_helper("False negative rate (FN / (FN + TP)): ", FN, FN + TP)
    print_stats_helper("F1 score (2TP / (2TP + FN + FP)): ", 2 * TP, 2 * TP + FN + FP)


def main():
    # Include all the features we want
    processFuncs = {}

    processFuncs['smart_193_raw'] = lambda x: 1 if x != '' and int(x) > 0 else 0
    processFuncs['smart_194_raw'] = lambda x: 1 if x != '' and int(x) > 0 else 0
    processFuncs['smart_241_raw'] = lambda x: 1 if x != '' and int(x) > 0 else 0
    processFuncs['smart_197_raw'] = lambda x: 1 if x != '' and int(x) > 0 else 0
    processFuncs['smart_9_raw'] = lambda x: 1 if x != '' and int(x) > 0 else 0
    

    folders = ["../test_10/", "../test_10/", "../test_10"]
    years = [2017, 2017, 2017]
    months = [[1,2,3], [4,5,6], [7,8,9]]
    days = [[31,28,31], [30,31,30], [31,31,30]]

    files = []
    for i in range(0, len(folders)):
        files += generateFileNames(folders[i], years[i], months[i], days[i])
        break

    
    # Extract data from files
    dataPointsX, dataPointsY, serialNumberToData = import_data.import_data_with_processing(files, processFuncs, set([]))
    
    # Use balanced data
    balancedX, balancedY = import_data.extract_balanced_data(dataPointsX, dataPointsY)

    print "Total number of points is: " + str(len(dataPointsY))
    print "Total failures is: " + str(sum(dataPointsY))


    print "Total disks in balanced set: %d, num of failures in balanced: %d" % (len(balancedY), sum(balancedY))
    model = None
    if model_to_use == 'logistic':
        model = trainLogisticRegression(balancedX, balancedY)
        print "Coefficients: " + str(model.coef_)
    elif model_to_use == 'random':
        model = trainRandomForest(balancedX, balancedY)
        print "Coefficients: " + str(model.feature_importances_)
    else:
        print "Unknown model.  Exiting..."
        return

    files = generateFileNamesNoTest("../test_final/", 2016, [4,5,6], [30,31,30])
    dataPointsX, dataPointsY, serialNumberToData = import_data.import_data_with_processing(files, processFuncs, set())

    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0
    print "Len of keys: " + str(len(serialNumberToData.keys()))
    numFailures = 0
    numToClassify = 0
    
    actual = []
    predicted = []
    for serialNumber in serialNumberToData.keys():
        highestScore = 0.0
        if numToClassify % 1000 == 0:
            print "Hit num %d" % numToClassify
        numToClassify += 1
        result = 0
        trueFailure = 0
        misClaffMax = 10
        misClaffPoints = []
        badDate = None
        for date in serialNumberToData[serialNumber].keys():
            dataPointX, dataPointY = serialNumberToData[serialNumber][date]
            if dataPointY > 0:
                trueFailure = 1
            tempResult = model.predict([dataPointX])[0]
            tempScore = model.predict_proba([dataPointX])[0][1]  # 1 is the probability of failure
            result += tempResult
            if trueFailure == 0 and result > 0 or trueFailure == 1 and result == 0:
                badDate = date
            
            if tempScore > highestScore:
                highestScore = tempScore
        
        numFailures += trueFailure
        if result >= 1:
            result = 1
        else:
            result = 0
        if result == trueFailure:
            if result == 1:
                truePos += 1
            else:
                trueNeg += 1
        else:
            if result == 1:
                falsePos += 1
            else:
                falseNeg += 1
            dataPointX, dataPointY = serialNumberToData[serialNumber][date]
        
        actual.append(trueFailure)
        predicted.append(highestScore)
        
    print_stats(float(truePos), float(trueNeg), float(falsePos), float(falseNeg))
    generate_roc(actual, predicted)
    
    actual_str = [str(actual[i]) for i in range(0, len(actual))]
    predicted_str = [str(predicted[i]) for i in range(0, len(predicted))]
    
    with open("logistic_actual_predicted_true_test.csv", "w+") as out_data:
        out_data.write(','.join(actual_str) + '\n')
        out_data.write(','.join(predicted_str) + '\n')

if __name__ == "__main__":
    main()
