print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import csv
import json

total_failure = 0

HDD_dict = {}  # key = serial num, value = vector of 5 raw values and failure value
days = [31,29, 31, 30, 31, 30] #1,2,3
for m in xrange(4,7):	
	for d in range(1,days[m-1] + 1):

		name_i = '0' + str(d) if d < 10 else str(d)
		file_data_name = "../test_data/2016-0{0}-{1}_smoothed_0.csv".format(m, name_i)
		print file_data_name
		
		# open each file and read specific attributes
		with open(file_data_name, 'r') as file_data:
			reader = csv.reader(file_data, delimiter='\t')
			data_list = list(reader)

			actual_data_list = data_list[1:] # contains all other data after first row
			attributes_list = data_list[0][0].split(',') # contains the headers and attribute names 
			
			# check index values for all attributes we want
			important_attributes = [ 'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw' ]

			for i, value in enumerate(actual_data_list):
				
				hard_drive_data = actual_data_list[i][0].split(',') # individual hard drive's data
				serial_number = hard_drive_data[1]

				attribute_dict = {}
				# attribute dict contains
				# 'raw values', 'failed'
				if serial_number in HDD_dict:
					attribute_dict = HDD_dict[serial_number]

				# set failure status
				if 'failed' not in attribute_dict:
					attribute_dict['failed'] = int(hard_drive_data[4])
				else: 
					if attribute_dict['failed'] != 1:
						attribute_dict['failed'] = int(hard_drive_data[4])

				total_failure += int(hard_drive_data[4])

				attribute_dict['smart_5_raw'] = int(hard_drive_data[14]) if hard_drive_data[14] != '' else None
				attribute_dict['smart_187_raw'] = int(hard_drive_data[38]) if hard_drive_data[38] != '' else None
				attribute_dict['smart_188_raw'] = int(hard_drive_data[40]) if hard_drive_data[40] != '' else None
				attribute_dict['smart_197_raw'] = int(hard_drive_data[58]) if hard_drive_data[58] != '' else None
				attribute_dict['smart_198_raw'] = int(hard_drive_data[60]) if hard_drive_data[60] != '' else None

				# Get total value of all the attributes
				total = 0
				total += attribute_dict['smart_5_raw'] if attribute_dict['smart_5_raw'] != None else 0
				total += attribute_dict['smart_187_raw'] if attribute_dict['smart_187_raw'] != None else 0
				total += attribute_dict['smart_188_raw'] if attribute_dict['smart_188_raw'] != None else 0
				total += attribute_dict['smart_197_raw'] if attribute_dict['smart_197_raw'] != None else 0
				total += attribute_dict['smart_198_raw'] if attribute_dict['smart_198_raw'] != None else 0

				# check if we have already predicted failure for this hard drive 
				if 'prediction' not in attribute_dict:
					attribute_dict['prediction'] = 0

				already_predicted_failture = attribute_dict['prediction'] == 1

				if (not already_predicted_failture):
					if total > 0:
						attribute_dict['prediction'] = 1
					else:
						attribute_dict['prediction'] = 0

				HDD_dict[serial_number] = attribute_dict

# Get information from each HDD

false_pos = 0 
false_neg = 0
true_pos = 0
true_neg = 0

actual =[]
predictions = []

for key in HDD_dict:
	disk = HDD_dict[key]
	actual.append(disk['failed'])
	predictions.append(disk['prediction'])

	if disk['prediction'] == 1 and disk['failed'] == 0:
		false_pos +=1
	if disk['prediction'] == 0 and disk['failed'] == 1:
		false_neg +=1
	if disk['prediction'] == 0 and disk['failed'] == 0:
		true_neg += 1
	if disk['prediction'] == 1 and disk['failed'] == 1:
		true_pos += 1

PPV = float(true_pos)/ (true_pos + false_pos)
NPV = float(true_neg)/ (false_neg + true_neg)
Accuracy= (float(true_pos) + true_neg)/ (true_pos + false_pos+ false_neg + true_neg)
Sensitivity = float(true_neg)/(false_pos + true_neg)
Specificity = float(true_pos)/(true_pos + false_neg)

print '--- RAW values ----'
print "False positive: {0}".format(false_pos)
print "True positive:{0}".format(true_pos)
print "False negative: {0}".format(false_neg)
print "True negative: {0}".format(true_neg)
print '--- RATES ----'
print'PPV', PPV
print'NPV', NPV
print 'Accuracy', Accuracy
print'Sensitivity', Sensitivity
print 'Specificity', Specificity 
print '------ ROC ------'
print 'Precision', float(true_pos)/(true_pos + false_pos)
print 'Recall', float(true_pos)/(true_pos+ false_neg)

false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)

# REPLACE: more descriptive title
plt.title('Receiver Operating Characteristic - Baseline')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
