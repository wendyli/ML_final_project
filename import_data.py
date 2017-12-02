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

def smooth_potential_failures_and_write(filenames, days, out_dir):
    true_failures = {}
    total_days = len(filenames)
    for index in range(0, total_days):
        filename = filenames[total_days - 1 - index]
        filename_stripped = filename.split('/')[-1].split('.')[0]
        with open(filename, 'r') as file_data, open(out_dir + filename_stripped + "_smoothed_" + str(days) + ".csv", 'w+') as out_data:
            reader = csv.reader(file_data, delimiter=',')
            data_list = list(reader)

            actual_data_list = data_list[1:]
            attributes_list = data_list[0]

            num_features = len(attributes_list)
            num_examples = len(actual_data_list)
            
            out_data.write(','.join(attributes_list) + '\n')
            for i in range(0, num_examples):
                ith_raw_data = actual_data_list[i]
                ith_data_vector = []
                ith_y_value = -1
                
                serialNumber = ''
                failure_index = -1
                for j in range(0, num_features):
                    attribute_name = attributes_list[j]
                    if attribute_name == 'failure':
                        ith_y_value = int(ith_raw_data[j])
                        failure_index = j
                    if attribute_name == 'serial_number':
                        serialNumber = ith_raw_data[j]
                    if ith_y_value >= 0 and serialNumber != '':
                        break
                
                if ith_y_value == 1:
                    true_failures[serialNumber] = total_days - 1 - index
                
                last_known_failure = true_failures.get(serialNumber, -1)
                if total_days - 1 - index <= last_known_failure and last_known_failure <= total_days - 1 - index + days:
                    print "serial number: " + serialNumber + " has failure"
                    print "index: " + str(index) + ", day: " + str(total_days - 1 - index) + ", last known failure: " + str(last_known_failure) + ", smoothing_days: " + str(days)
                    ith_raw_data[failure_index] = '1'
                out_data.write(','.join(ith_raw_data) + '\n')


def generateFileNames(dir, year, months, days):
    filenames = []
    yearStr = str(year)
    for i in range(0, len(months)):
        monthStr = '0' + str(months[i]) if months[i] < 10 else str(months[i])
        for j in range(1, days[i]+1):
            dayStr = '0' + str(j) if j < 10 else str(j)
            file_data_name = dir + "{0}-{1}-{2}.csv".format(yearStr, monthStr, dayStr)
            filenames.append(file_data_name)
    
    return filenames


def main():
    smoothing_days = 60
    folders = ["../data_2016_Q1/", "../data_2016_Q2/", "../data_2016_Q3/", "../data_2016_Q4/", "../data_2017_Q1/", "../data_2017_Q2/", "../data_2017_Q3/"]
    years = [2016, 2016, 2016, 2016, 2017, 2017, 2017]
    months = [[1,2,3], [4,5,6], [7,8,9], [10,11,12], [1,2,3], [4,5,6], [7,8,9]]
    days = [[31,29,31], [30,31,30], [31,31,30], [31,30,31], [31,28,31], [30,31,30], [31,31,30]]
    files = []
    for i in range(0, len(folders)):
        files += generateFileNames(folders[i], years[i], months[i], days[i])
    
    print files
    smooth_potential_failures_and_write(files, smoothing_days, "../test/")


if __name__ == "__main__":
    main()