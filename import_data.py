import csv
import numpy
import collections

# Parameters:
# - filenames: a list of paths to files to parse
# - filter: a set of attributes to look for
# - include: True if we should only include features that are in the filter,
#            and False if we should exclude features that are in the filter.
#
# Returns tuple consisting of the following:
# - a vector of the disk data where each data point is transformed into a vector of real numbers
# - a vector of 1s and 0s (dataPointsY[i] is 1 if disk i failed and 0 otherwise)
# - a vector of serial numbers (serialNumber[i] is the serial number of disk i; several entries may
#   be the same, which means that such entries correspond to the same disk at different time points)
# Note: defaults to including all features except for 'data', 'serial_number', and 'model'.
def import_data(filenames, filter=set(['date', 'serial_number', 'model']), include=False):
    dataPointsX = []
    dataPointsY = []
    serialNumberToData = collections.defaultdict(list)
    for filename in filenames:
        with open(filename, 'r') as file_data:
            reader = csv.reader(file_data, delimiter=',')
            data_list = list(reader)

            actual_data_list = data_list[1:]
            attributes_list = data_list[0]

            num_features = len(attributes_list)
            num_examples = len(actual_data_list)
            
            for i in range(0, num_examples):
                ith_raw_data = actual_data_list[i]
                ith_data_vector = []
                ith_y_value = -1
                
                serialNumber = ''
                for j in range(0, num_features):
                    attribute_name = attributes_list[j]
                    in_filter = attribute_name in filter
                    if in_filter == include:
                        if ith_raw_data[j] != '':
                            ith_data_vector.append(int(ith_raw_data[j]))
                        else:
                            ith_data_vector.append(0)
                    if attribute_name == 'failure':
                        ith_y_value = int(ith_raw_data[j])
                    if attribute_name == 'serial_number':
                        serialNumber = ith_raw_data[j]
                
                dataPointsX.append(ith_data_vector)
                dataPointsY.append(ith_y_value)
                serialNumberToData[serialNumber].append((ith_data_vector, ith_y_value))
                
    dataPointsX = numpy.array(dataPointsX)
    dataPointsY = numpy.array(dataPointsY)
    return dataPointsX, dataPointsY, serialNumberToData

# Same as import_data, but filter is now a dictionary of functions, where if include is True,
# the corresponding function is called to process the given feature.
#
#
# Parameters:
# - filenames: a list of paths to files to parse
# - processFuncs: a map from feature name to processing function to call (only features that are found in
#                 the map will be included)
# - filter: a set of y values that we should not include
# - maxPoints: maximum number of points to read in
#
# Returns tuple consisting of the following:
# - a vector of the disk data where each data point is transformed into a vector of real numbers
# - a vector of 1s and 0s (dataPointsY[i] is 1 if disk i failed and 0 otherwise)
# - a vector of serial numbers (serialNumber[i] is the serial number of disk i; several entries may
#   be the same, which means that such entries correspond to the same disk at different time points)
# Note: defaults to including all features except for 'data', 'serial_number', and 'model'.
def import_data_with_processing(filenames, processFuncs=dict(), filter=set(), maxPoints=None):
    dataPointsX = []
    dataPointsY = []
    serialNumberToData = collections.defaultdict(list)
    exceededMax = False
    for filename in filenames:
        with open(filename, 'r') as file_data:
            reader = csv.reader(file_data, delimiter=',')
            data_list = list(reader)

            actual_data_list = data_list[1:]
            attributes_list = data_list[0]

            num_features = len(attributes_list)
            num_examples = len(actual_data_list)
            
            for i in range(0, num_examples):
                if maxPoints is not None and maxPoints <= 0:
                    exceededMax = True
                    break
                ith_raw_data = actual_data_list[i]
                ith_data_vector = []
                ith_y_value = -1
                
                serialNumber = ''
                for j in range(0, num_features):
                    attribute_name = attributes_list[j]
                    if attribute_name in processFuncs:
                        ith_data_vector.append(processFuncs[attribute_name](ith_raw_data[j]))
                    if attribute_name == 'failure':
                        ith_y_value = int(ith_raw_data[j])
                    if attribute_name == 'serial_number':
                        serialNumber = ith_raw_data[j]
                
                if ith_y_value not in filter:
                    if maxPoints is not None:
                        maxPoints -= 1
                    dataPointsX.append(ith_data_vector)
                    dataPointsY.append(ith_y_value)
                    serialNumberToData[serialNumber].append((ith_data_vector, ith_y_value))
            
            if exceededMax:
                break
                
    dataPointsX = numpy.array(dataPointsX)
    dataPointsY = numpy.array(dataPointsY)
    return dataPointsX, dataPointsY, serialNumberToData