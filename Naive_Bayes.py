import import_data
import numpy as np
import collections
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

featureY = "failure"
filenameDirectory = "../test_10/"


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

# Train Multinomial Naive-Bayes classification model on disk dataset
# Parameters:
#   data: (featureNames, map of serial number to disk data point)
#   features: set of features to train on
#
# Returns (mapping_from_attribute_name_to_int, trained_regression_model)
def trainNB(dataPointsX, dataPointsY):
    mnb = MultinomialNB()
    mnb.fit(dataPointsX, np.ravel(dataPointsY))
    return mnb


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


def bucket_smart_5_raw(x):
    if x == '':
        return 0

    intX = int(x)
    if (intX >= 0 and intX < 7):
        return 0
    elif (intX >= 8 and intX < 150):
        return 50
    else:
        return 1000

def bucket_smart_198_raw(x):
    if x == '':
        return 0

    intX = int(x)
    if (intX >= 0 and intX < 16):
        return 0
    elif (intX >= 16 and intX < 24):
        return 16
    elif (intX >= 24 and intX < 32):
        return 24
    elif (intX >= 32 and intX < 150):
       return 100
    else:
       return 1000
   

def bucket_smart_188_raw(x):
    if x == '':
        return 0

    intX = int(x)
    if (intX >= 0 and intX < 16):
        return 0
    elif (intX >= 16 and intX < 24):
        return 16
    elif (intX >= 24 and intX < 32):
        return 24
    elif (intX >= 32 and intX < 150):
       return 100
    else:
       return 1000
      
# Include all the features we want
processFuncs = {}
#processFuncs['smart_5_raw'] = process_smart_5_raw
processFuncs['smart_5_raw'] = lambda x: int(x) if x != '' and int(x) > 0 else 0
processFuncs['smart_198_raw'] = lambda x: int(x) if x != '' and int(x) > 0 else 0
#processFuncs['smart_187_raw'] = process_smart_187_raw
processFuncs['smart_187_raw'] = lambda x: int(x) if x != '' and int(x) > 0 else 0
#processFuncs['smart_188_raw'] = process_smart_188_raw
processFuncs['smart_188_raw'] = lambda x: int(x) if x != '' and int(x) > 0 else 0
#processFuncs['smart_197_raw'] = lambda x: int(x) if x != '' else 0
processFuncs['smart_197_raw'] = lambda x: int(x) if x != '' and int(x) > 0 else 0

    
#folders = ["../data_Q1_2016/", "../data_Q2_2016/", "../data_Q3_2016/", "../data_Q4_2016/", "../data_Q1_2017/", "../data_Q2_2017/", "../data_Q3_2017/"]
#folders = ["../test_10/", "../test_10/", "../test_10/", "../test_10/", "../test_10/", "../test_10/", "../test_10/"]
#years = [2016, 2016, 2016, 2016, 2017, 2017, 2017]
#months = [[1,2,3], [4,5,6], [7,8,9], [10,11,12], [1,2,3], [4,5,6], [7,8,9]]
#days = [[31,29,31], [30,31,30], [31,31,30], [31,30,31], [31,28,31], [30,31,30], [31,31,30]]


folders = ["../test_10/", "../test_10/"]
years = [2017, 2017]
months = [[1,2,3], [4,5,6]]
days = [[31,28,31], [30,31,30]]

files = []
for i in range(0, len(folders)):
    files += generateFileNames(folders[i], years[i], months[i], days[i])


#filenameDirectory = "../data_Q2_2017/"
#files = files + generateFileNames([4, 5, 6], [30, 31, 30])
# dataPointsX, dataPointsY, serialNumberToData = import_data.import_data(files, filter=set(['smart_1_raw', 'smart_3_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_10_raw', 'smart_12_raw', 'smart_184_raw', 'smart_187_raw', 'smart_188_raw', 'smart_192_raw', 'smart_193_raw', 'smart_194_raw', 'smart_196_raw', 'smart_197_raw', 'smart_198_raw', 'smart_199_raw']), include=True)
dataPointsX, dataPointsY, serialNumberToData = import_data.import_data_with_processing(files, processFuncs, set([]))
#dataPointsX2, dataPointsY2, serialNumberToData2 = import_data.import_data_with_processing(files, processFuncs, set([1]), len(dataPointsY))

'''
dataPointsX = numpy.concatenate((dataPointsX, dataPointsX2))
dataPointsY = numpy.concatenate((dataPointsY, dataPointsY2))
for serialNumber in serialNumberToData2:
    points = serialNumberToData.get(serialNumber, []) + serialNumberToData2[serialNumber]
    serialNumberToData[serialNumber] = points
'''
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
tempX = []
tempY = []
max1 = 0
max0 = 0
maxVal = 3620
for i in range(0, len(dataPointsX)):
    missing = False
    '''
    for j in range(0, len(dataPointsX[i])):
        if dataPointsX[i][j] < 0:
            missing = True
            break
    if missing:
        continue
    '''
    if dataPointsY[i] == 0 and max0 < maxVal:
        tempX.append(dataPointsX[i])
        tempY.append(dataPointsY[i])
        max0 += 1
    if dataPointsY[i] == 1 and max1 < maxVal:
        tempX.append(dataPointsX[i])
        tempY.append(dataPointsY[i])
        max1 += 1

dataPointsX = tempX
dataPointsY = tempY
    
print "Total disks: %d, num of failures: %d" % (len(dataPointsY), sum(dataPointsY))
reg = trainNB(dataPointsX, dataPointsY)
print "Coefficients: " + str(reg.coef_)


files = generateFileNamesNoTest("../data_2016_Q2/", 2016, [4,5,6], [30,31,30])
dataPointsX, dataPointsY, serialNumberToData = import_data.import_data_with_processing(files, processFuncs, set())

truePos = 0
falsePos = 0
trueNeg = 0
falseNeg = 0


print "Len of keys: " + str(len(serialNumberToData.keys()))
numFailures = 0

actualVec = []
predictionsVec = []

for serialNumber in serialNumberToData.keys():
    result = 0
    trueFailure = 0
    misClaffMax = 10
    misClaffPoints = []
    badDate = None

    allProbsVec = []
    failProb = 0.0
    for date in serialNumberToData[serialNumber].keys():
        dataPointX, dataPointY = serialNumberToData[serialNumber][date]
        
        if dataPointY > 0:
            trueFailure = 1

        tempResult = reg.predict([dataPointX])[0]
        
        allProbsVec.append( reg.predict_proba([dataPointX])[0,1] )
                           
        result += tempResult
        if trueFailure == 0 and result > 0 or trueFailure == 1 and result == 0:
            badDate = date

    maxProbPredicted = max(allProbsVec)

    actualVec.append(trueFailure)
    predictionsVec.append(maxProbPredicted)
    
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
        #dataPointX, dataPointY = serialNumberToData[serialNumber][date]
        #print "Misclassified date: %s, pointX: %s, pointY: %s, predicted value: %s, true value: %s" % (str(badDate), str(dataPointX), str(dataPointY), str(result), str(trueFailure))

                        
print "True positive: " + str(truePos)
print "False positive: " + str(falsePos)
print "True negative: " + str(trueNeg)
print "False negative: " + str(falseNeg)

print filenameDirectory


print "True positive rate (TP / (TP + FN)): " + str(float(truePos) / (truePos + falseNeg))
print "True negative rate (TN / (TN + FP)): " + str(float(trueNeg) / (trueNeg + falsePos))
print "False positive rate (FP / (FP + TN)): " + str(float(falsePos) / (falsePos + trueNeg))
print "False negative rate (FN / (FN + TP)): " + str(float(falseNeg) / (falseNeg + truePos))

false_positive_rate, true_positive_rate, thresholds = roc_curve(actualVec, predictionsVec)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic - Naive Bayes')
plt.plot(false_positive_rate, true_positive_rate, 'b',
         label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
show()
