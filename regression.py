import import_data
import numpy
import collections
import sys
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

featureY = "failure"
filenameDirectory = "../data_Q1_2017/"


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
            file_data_name = dir + "{0}-{1}-{2}.csv".format(yearStr, monthStr, dayStr)
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


# Assumes the arguments are floats
def print_stats(TP, TN, FP, FN):
    print "True positive: " + str(TP)
    print "True negative: " + str(TN)
    print "False positive: " + str(FP)
    print "False negative: " + str(FN)

    print_stats_helper("True positive rate (TP / (TP + FP)): ", TP, TP + FP)
    print_stats_helper("True negative rate (TN / (TN + FN)): ", TN, TN + FN)
    print_stats_helper("False positive rate (FP / (FP + TN)): ", FP, FP + TN)
    print_stats_helper("False negative rate (FN / (FN + TP)): ", FN, FN + TP)
    print_stats_helper("F1 score (TP / (2TP + FN + FP)): ", TP, 2 * TP + FN + FP)


def main():
    # Include all the features we want
    processFuncs = {}
    #processFuncs['smart_5_raw'] = process_smart_5_raw
    #processFuncs['smart_5_raw'] = lambda x: int(x) if x != '' and int(x) > 0 else 0
    #processFuncs['smart_187_raw'] = process_smart_187_raw
    #processFuncs['smart_187_raw'] = lambda x: int(x) if x != '' and int(x) > 0 else 0
    #processFuncs['smart_188_raw'] = process_smart_188_raw
    #processFuncs['smart_188_raw'] = lambda x: int(x) if x != '' and int(x) > 0 else 0
    #processFuncs['smart_197_raw'] = lambda x: int(x) if x != '' else 0
    #processFuncs['smart_197_raw'] = lambda x: int(x) if x != '' and int(x) > 0 else 0
    #processFuncs['smart_198_raw'] = process_smart_198_raw
    #processFuncs['smart_198_raw'] = lambda x: int(x) if x != '' and int(x) > 0 else 0

    processFuncs['smart_193_raw'] = lambda x: 1 if x != '' and int(x) > 0 else 0
    processFuncs['smart_194_raw'] = lambda x: 1 if x != '' and int(x) > 0 else 0
    processFuncs['smart_241_raw'] = lambda x: 1 if x != '' and int(x) > 0 else 0
    processFuncs['smart_197_raw'] = lambda x: 1 if x != '' and int(x) > 0 else 0
    processFuncs['smart_9_raw'] = lambda x: 1 if x != '' and int(x) > 0 else 0

    
    # Set up the files we wish to use
    #folders = ["../data_Q1_2016/", "../data_Q2_2016/", "../data_Q3_2016/", "../data_Q4_2016/", "../data_Q1_2017/", "../data_Q2_2017/", "../data_Q3_2017/"]
    #folders = ["../test_10/", "../test_10/", "../test_10/", "../test_10/", "../test_10/", "../test_10/", "../test_10/"]
    #years = [2016, 2016, 2016, 2016, 2017, 2017, 2017]
    #months = [[1,2,3], [4,5,6], [7,8,9], [10,11,12], [1,2,3], [4,5,6], [7,8,9]]
    #days = [[31,29,31], [30,31,30], [31,31,30], [31,30,31], [31,28,31], [30,31,30], [31,31,30]]

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
    
    # Use balanced data if desired
    # balancedX = dataPointsX
    # balancedY = dataPointsY
    balancedX, balancedY = import_data.extract_balanced_data(dataPointsX, dataPointsY)

    print "Total number of points is: " + str(len(dataPointsY))
    print "Total failures is: " + str(sum(dataPointsY))

    '''
    totalUnique = 0
    for serialNumber in serialNumberToData.keys():
        for (dataPointX, dataPointY) in serialNumberToData[serialNumber]:
            if dataPointY > 0:
                #print serialNumber
                totalUnique += 1
                break
                
                
    print "Total unique failures: " + str(totalUnique)
    '''
        
    print "Total disks in balanced set: %d, num of failures in balanced: %d" % (len(balancedY), sum(balancedY))
    model = trainLogisticRegression(balancedX, balancedY)
    print "Coefficients: " + str(model.coef_)
    # model = trainRandomForest(balancedX, balancedY)
    #print "Coefficients: " + str(reg.feature_importances_)

    files = generateFileNamesNoTest("../data_2016_Q1/", 2016, [1,2,3], [31,29,31])
    dataPointsX, dataPointsY, serialNumberToData = import_data.import_data_with_processing(files, processFuncs, set())

    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0
    print "Len of keys: " + str(len(serialNumberToData.keys()))
    numFailures = 0
    numToClassify = 0
    for serialNumber in serialNumberToData.keys():
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
            result += tempResult
            if trueFailure == 0 and result > 0 or trueFailure == 1 and result == 0:
                badDate = date
        
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
            #print "Disk %s was misclassified." % serialNumber
            if result == 1:
                #print "False Positive"
                falsePos += 1
            else:
                #print "False Negative"
                falseNeg += 1
            dataPointX, dataPointY = serialNumberToData[serialNumber][date]
            #print "Misclassified date: %s, pointX: %s, pointY: %s, predicted value: %s, true value: %s" % (str(badDate), str(dataPointX), str(dataPointY), str(result), str(trueFailure))
    
    print_stats(float(truePos), float(trueNeg), float(falsePos), float(falseNeg))

if __name__ == "__main__":
    main()
