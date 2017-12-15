import import_data
import numpy
import collections
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

print(__doc__)
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import csv
import json

featureY = "failure"

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
# Returns (mapping_from_attribute_name_to_int, trained_regression_model)
def trainModel(dataPointsX, dataPointsY):
    #reg = linear_model.LogisticRegression(class_weight='balanced')
    #reg = linear_model.LogisticRegression()
    #reg = RandomForestClassifier(max_depth=2, random_state=0)
    reg = RandomForestClassifier(
            bootstrap=True, 
            class_weight={1: 1.1, 0:1},# increase weight of minority class 
            criterion='gini',
            max_depth=1, 
            max_features='auto', 
            max_leaf_nodes=None,
            min_impurity_decrease=0.0, 
            min_impurity_split=None,
            min_samples_leaf=40, 
            min_samples_split=2,
            min_weight_fraction_leaf=1e-3, 
            n_estimators=10, 
            n_jobs=1,
            oob_score=False, 
            random_state=0, 
            verbose=0, 
            warm_start=False)

    reg.fit(dataPointsX, dataPointsY)
    return reg

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

# Include all the features we want
processFuncs = {}

processFuncs['smart_198_raw'] = lambda x: int(x) if x != '' else 0
processFuncs['smart_197_raw'] = lambda x: int(x) if x != '' else 0
processFuncs['smart_188_raw'] = lambda x: int(x) if x != '' else 0
processFuncs['smart_187_raw'] = lambda x: int(x) if x != '' else 0 # not useful 
processFuncs['smart_5_raw'] = lambda x: int(x) if x != '' else 0
processFuncs['smart_193_raw'] = lambda x: int(x) if x != '' else 0 # not useful
processFuncs['smart_194_raw'] = lambda x: int(x) if x != '' else 0
processFuncs['smart_241_raw'] = lambda x: int(x) if x != '' else 0
processFuncs['smart_9_raw'] = lambda x: int(x) if x != '' else 0 # not useful
processFuncs['smart_242_raw'] = lambda x: int(x) if x != '' else 0
#folders = ["../data_Q1_2016/", "../data_Q2_2016/", "../data_Q3_2016/", "../data_Q4_2016/", "../data_Q1_2017/", "../data_Q2_2017/", "../data_Q3_2017/"]
#folders = ["../test_10/", "../test_10/", "../test_10/", "../test_10/", "../test_10/", "../test_10/", "../test_10/"]
#years = [2016, 2016, 2016, 2016, 2017, 2017, 2017]
#months = [[1,2,3], [4,5,6], [7,8,9], [10,11,12], [1,2,3], [4,5,6], [7,8,9]]
#days = [[31,29,31], [30,31,30], [31,31,30], [31,30,31], [31,28,31], [30,31,30], [31,31,30]]

folders = ["../test_10/", "../test_10/", "../test_10/"]
years = [2017, 2017, 2017]
months = [[1,2,3], [4,5,6], [7,8,9]]
days = [[31,28,31], [30,31,30], [31,31,30]]

files = []
for i in range(0, len(folders)):
    files += generateFileNames(folders[i], years[i], months[i], days[i])
    break

#filenameDirectory = "../data_Q2_2017/"
#files = files + generateFileNames([4, 5, 6], [30, 31, 30])
# dataPointsX, dataPointsY, serialNumberToData = import_data.import_data(files, filter=set(['smart_1_raw', 'smart_3_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_10_raw', 'smart_12_raw', 'smart_184_raw', 'smart_187_raw', 'smart_188_raw', 'smart_192_raw', 'smart_193_raw', 'smart_194_raw', 'smart_196_raw', 'smart_197_raw', 'smart_198_raw', 'smart_199_raw']), include=True)
dataPointsX, dataPointsY, serialNumberToData = import_data.import_data_with_processing(files, processFuncs, set([]))
#dataPointsX2, dataPointsY2, serialNumberToData2 = import_data.import_data_with_processing(files, processFuncs, set([1]), len(dataPointsY))

balancedX, balancedY = import_data.extract_balanced_data(dataPointsX, dataPointsY)

print "Total number of points is: " + str(len(dataPointsY))
print "Total failures is: " + str(sum(dataPointsY))

# ---------------------------

print "Total disks: %d, num of failures: %d" % (len(balancedY), sum(balancedY))
reg = trainModel(balancedX, balancedY)
#print "Coefficients: " + str(reg.coef_)
print "Coefficients: " + str(reg.feature_importances_)

files = generateFileNamesNoTest("../data_2016_Q1/", 2016, [1,2,3], [31,29,31])
#files = generateFileNamesNoTest("../data_2016_Q1/", 2016, [1], [3])
dataPointsX, dataPointsY, serialNumberToData = import_data.import_data_with_processing(files, processFuncs, set())

truePos = 0
falsePos = 0
trueNeg = 0
falseNeg = 0
print "Len of keys: " + str(len(serialNumberToData.keys()))
numFailures = 0
numToClassify = 0

# For ROC
actual = []
predictions = []
prob_predictions = []

for serialNumber in serialNumberToData.keys():
    if numToClassify % 1000 == 0:
        print "Hit num %d" % numToClassify
    numToClassify += 1
    result = 0
    trueFailure = 0
    misClaffMax = 10
    misClaffPoints = []
    badDate = None

    maxProb = 0 # probability of failure
    for date in serialNumberToData[serialNumber].keys():
        dataPointX, dataPointY = serialNumberToData[serialNumber][date]
        if dataPointY > 0:
            trueFailure = 1
        tempResult = reg.predict([dataPointX])[0]
        tempProb = reg.predict_proba([dataPointX])[0][1]
        result += tempResult
        maxProb = max(tempProb, maxProb)
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
    # For ROC
    predictions.append(int(result))
    actual.append(int(trueFailure))
    prob_predictions.append(int(maxProb))

print "True positive: " + str(truePos)
print "False positive: " + str(falsePos)
print "True negative: " + str(trueNeg)
print "False negative: " + str(falseNeg)

print "True positive rate (TP / (TP + FN)): " + str(float(truePos) / (truePos + falseNeg))
print "True negative rate (TN / (TN + FP)): " + str(float(trueNeg) / (trueNeg + falsePos))
print "False positive rate (FP / (FP + TN)): " + str(float(falsePos) / (falsePos + trueNeg))
print "False negative rate (FN / (FN + TP)): " + str(float(falseNeg) / (falseNeg + truePos))

#----------- Generate Random Forest ----------------

false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, prob_predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)

# ---------- Check ROC -----------------
# save to excel sheet
zipped = zip(actual, predictions) 
#np.savetxt('RF_ROC.csv', zipped, fmt='%i,%i', header= 'actual - RF')

# REPLACE: more descriptive title
plt.title('Receiver Operating Characteristic - Random Forest')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


