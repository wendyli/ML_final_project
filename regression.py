import import_data
import numpy
import collections
from sklearn import linear_model

featureY = "failure"
filenameDirectory = "../data_Q1_2017/"


def generateFileNames(months, days):
    filenames = []
    for i in range(0, len(months)):
        monthStr = '0' + str(months[i]) if months[i] < 10 else str(months[i])
        for j in range(1, days[i]+1):
            dayStr = '0' + str(j) if j < 10 else str(j)
            file_data_name = filenameDirectory + "2017-{0}-{1}.csv".format(monthStr, dayStr)
            filenames.append(file_data_name)
    
    return filenames


# Parameters:
#   data: (featureNames, map of serial number to disk data point)
#   features: set of features to train on
#
# Returns (mapping_from_attribute_name_to_int, trained_regression_model)
def trainLogisticRegression(dataPointsX, dataPointsY):
    reg = linear_model.LogisticRegression()
    reg.fit(dataPointsX, dataPointsY)
    return reg


files = generateFileNames([1, 2, 3], [31, 28, 31])
dataPointsX, dataPointsY, serialNumberToData = import_data.import_data(files, filter=set(['smart_1_raw', 'smart_3_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_10_raw', 'smart_12_raw', 'smart_184_raw', 'smart_187_raw', 'smart_188_raw', 'smart_192_raw', 'smart_193_raw', 'smart_194_raw', 'smart_196_raw', 'smart_197_raw', 'smart_198_raw', 'smart_199_raw']), include=True)
print "Total failures is: " + str(sum(dataPointsY))

totalUnique = 0
for serialNumber in serialNumberToData.keys():
    for (dataPointX, dataPointY) in serialNumberToData[serialNumber]:
        if dataPointY > 0:
            print serialNumber
            totalUnique += 1
            break
            
            
print "Total unique failures: " + str(totalUnique)
tempX = []
tempY = []
max1 = 0
max0 = 0
max = 3000
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
    if dataPointsY[i] == 0 and max0 < max:
        tempX.append(dataPointsX[i])
        tempY.append(dataPointsY[i])
        max0 += 1
    if dataPointsY[i] == 1 and max1 < max:
        tempX.append(dataPointsX[i])
        tempY.append(dataPointsY[i])
        max1 += 1

dataPointsX = tempX
dataPointsY = tempY
    
print "Total disks: %d, num of failures: %d" % (len(dataPointsY), sum(dataPointsY))
reg = trainLogisticRegression(dataPointsX, dataPointsY)
print "Coefficients: " + str(reg.coef_)

truePos = 0
falsePos = 0
trueNeg = 0
falseNeg = 0
print "Len of keys: " + str(len(serialNumberToData.keys()))
numFailures = 0
for serialNumber in serialNumberToData.keys():
    result = 0
    trueFailure = 0
    for (dataPointX, dataPointY) in serialNumberToData[serialNumber]:
        if dataPointY > 0:
            trueFailure = 1
        tempResult = reg.predict([dataPointX])[0]
        if tempResult > 0:
            result = tempResult
    numFailures += trueFailure
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
        
print "True positive: " + str(truePos)
print "False positive: " + str(falsePos)
print "True negative: " + str(trueNeg)
print "False negative: " + str(falseNeg)


print "True positive rate (TP / (TP + FP)): " + str(float(truePos) / (truePos + falsePos))
print "False negative rate (FN / (FN + TP)): " + str(float(falseNeg) / (falseNeg + truePos))
