import import_data
import numpy
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
dataPointsX, dataPointsY = import_data.import_data(files, filter=set(['smart_9_raw', 'smart_10_raw']), include=True)
print "Total disks: %d, num of failures: %d" % (len(dataPointsY), sum(dataPointsY))
print len(dataPointsX)
print dataPointsX[0:20]
print len(dataPointsY)
print dataPointsY[0:20]
reg = trainLogisticRegression(dataPointsX, dataPointsY)

truePos = 0
falsePos = 0
trueNeg = 0
falseNeg = 0
for i in range(0, len(dataPointsX)):
    result = reg.predict([dataPointsX[i]])
    if result == dataPointsY[i]:
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
