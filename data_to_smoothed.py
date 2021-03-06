#!/usr/bin/python

import csv
import numpy
import collections


def smooth_potential_failures_and_write(filenames, days, out_dir, prefix=''):
    true_failures = {}  # Map from serial number to earliest day it failed
    total_days = len(filenames)
    len_prefix = len(prefix)
    for index in range(0, total_days):
        filename = filenames[total_days - 1 - index]  # Assumes files are in chronological order; work backwards
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
                model = ''
                failure_index = -1
                for j in range(0, num_features):
                    attribute_name = attributes_list[j]
                    if attribute_name == 'failure':
                        ith_y_value = int(ith_raw_data[j])
                        failure_index = j
                    if attribute_name == 'serial_number':
                        serialNumber = ith_raw_data[j]
                    if attribute_name == 'model':
                        model = ith_raw_data[j]
                    if ith_y_value >= 0 and serialNumber != '' and model != '':  # Did we get everything?
                        break
                
                if model != '' and model[0:len_prefix] != prefix:
                    continue
                
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
    smoothing_days = 10
    folders = ["../data_2017_Q1/", "../data_2017_Q2/", "../data_2017_Q3/"]  # Assumes folders are passed in chronological order
    years = [2017, 2017, 2017]
    months = [[1,2,3], [4,5,6], [7,8,9]]
    days = [[31,28,31], [30,31,30], [31,31,30]]
    files = []
    for i in range(0, len(folders)):
        files += generateFileNames(folders[i], years[i], months[i], days[i])
    
    print files
    smooth_potential_failures_and_write(files, smoothing_days, "../test_10/")


if __name__ == "__main__":
    main()