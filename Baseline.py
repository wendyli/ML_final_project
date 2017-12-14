print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc


import csv
import json

'''
Only for Baseline analysis of Q1 2016  

'''

'''
Dict format:

Key: serial number of the hard disk drive 

Dict:
0 - 'predicted_status' 0 = not failed, 1 = failed
1 - 'predicted_fail_date' -> if no fail predicted, value is None
2 - 'actual_status' 0 = not failed, 1 = failed
3 - 'actual_fail_date' -> if not failed, value is None
4 - 'smart_5_raw'  		i = 14
5 - 'smart_187_raw'		i = 38
6 - 'smart_188_raw'		i = 40
7 - 'smart_197_raw'		i = 58
8 - 'smart_198_raw'		i = 60

'''

		
SMART_dict = {}  # key = serial num

days = [31, 29, 31] #1,2,3

for m in xrange(1,4):

	day = days[m-1]	
	for i in range(1,day+1):

		name_i = '0' + str(i) if i < 10 else str(i)
		file_data_name = "../data_2016_Q1/2016-0{0}-{1}.csv".format(m, name_i)
		print file_data_name

		with open(file_data_name, 'r') as file_data:
			reader = csv.reader(file_data, delimiter='\t')
			data_list = list(reader)

			actual_data_list = data_list[1:] # contains all other data after first row
			attributes_list = data_list[0][0].split(',') # contains the headers and attribute names 
			
			attribute_size = len(attributes_list) # total number of attributes
			feature_count = len(actual_data_list) # total number of SMART features

			# check index values for all attributes we want
			important_attributes = [ 'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw' ]

			for i, value in enumerate(actual_data_list):
				
				hard_drive_data = actual_data_list[i][0].split(',') # individual hard drive's data
				
				serial_number = hard_drive_data[1]

				attribute_dict = {}
				if serial_number in SMART_dict:
					attribute_dict = SMART_dict[serial_number]

			
				attribute_dict['actual_status'] = int(hard_drive_data[4])
				#print attributes_list[4], ':', attribute_dict['actual_status']
				
				attribute_dict['date'] = hard_drive_data[0]
				if attribute_dict['actual_status'] == 1:
					actual_fail_date = attribute_dict['date'] ### MODIFY ONCE GOING THROUGH MULTIPLE FILES
				else: 
					actual_fail_date = None

				attribute_dict['actual_fail_date'] = actual_fail_date
				#print 'actual_fail_date:', actual_fail_date

				
				attribute_dict['smart_5_raw'] = int(hard_drive_data[14]) if hard_drive_data[14] != '' else None
				#print attributes_list[14], ':', hard_drive_data[14]
				attribute_dict['smart_187_raw'] = int(hard_drive_data[38]) if hard_drive_data[38] != '' else None
				#print attributes_list[38], ':', hard_drive_data[38]
				attribute_dict['smart_188_raw'] = int(hard_drive_data[40]) if hard_drive_data[40] != '' else None
				#print attributes_list[40], ':', hard_drive_data[40]
				attribute_dict['smart_197_raw'] = int(hard_drive_data[58]) if hard_drive_data[58] != '' else None
				#print attributes_list[58], ':', hard_drive_data[58]
				attribute_dict['smart_198_raw'] = int(hard_drive_data[60]) if hard_drive_data[60] != '' else None
				#print attributes_list[60], ':', hard_drive_data[60]

				# Get total value of all the attributes
				total = 0
				total += attribute_dict['smart_5_raw'] if attribute_dict['smart_5_raw'] != None else 0
				total += attribute_dict['smart_187_raw'] if attribute_dict['smart_187_raw'] != None else 0
				total += attribute_dict['smart_188_raw'] if attribute_dict['smart_188_raw'] != None else 0
				total += attribute_dict['smart_197_raw'] if attribute_dict['smart_197_raw'] != None else 0
				total += attribute_dict['smart_198_raw'] if attribute_dict['smart_198_raw'] != None else 0

				#check if we have already predicted failure for this hard drive 
				if 'status_predicted' not in attribute_dict:
					attribute_dict['status_predicted'] = 0

				already_predicted_failture = attribute_dict['status_predicted'] == 1

				if (not already_predicted_failture):
					if total > 0:
						attribute_dict['status_predicted'] = 1
						attribute_dict['predicted_fail_date'] = attribute_dict['date'] #Change 
					else:
						attribute_dict['status_predicted'] = 0
						attribute_dict['predicted_fail_date'] = None

				SMART_dict[serial_number] = attribute_dict

# Get information from each HDD

count_actual_failures = 0
count_actual_healthy = 0
count_predicted_failures = 0

false_pos = 0 
false_neg = 0
true_pos = 0
true_neg = 0

num_examples = len(SMART_dict)

actual =[]
predictions = []

for key in SMART_dict:
	curr_attribute_dict = SMART_dict[key]
	count_actual_failures += curr_attribute_dict['actual_status']
	count_actual_healthy += (int)(curr_attribute_dict['actual_status'] == 0)
	count_predicted_failures += curr_attribute_dict['status_predicted']
	actual.append(curr_attribute_dict['actual_status'])
	predictions.append(curr_attribute_dict['status_predicted'])

	if curr_attribute_dict['status_predicted'] == 1 and curr_attribute_dict['actual_status'] == 0:
		false_pos += 1
	elif curr_attribute_dict['status_predicted'] == 0 and curr_attribute_dict['actual_status'] == 1:
		false_neg += 1
	elif curr_attribute_dict['status_predicted'] == 0 and curr_attribute_dict['actual_status'] == 0:
		true_neg += 1
	elif curr_attribute_dict['status_predicted'] == 1 and curr_attribute_dict['actual_status'] == 1:
		true_pos += 1


PPV = float(true_pos)/ (true_pos + false_pos)
NPV = float(true_neg)/ (false_neg + true_neg)
Accuracy= (float(true_pos) + true_neg)/ (true_pos + false_pos+ false_neg + true_neg)
Sensitivity = float(true_neg)/(false_pos + true_neg)
Specificity = float(true_pos)/(true_pos + false_neg)

print 'True failures:', count_actual_failures
print 'True healthy:', count_actual_healthy
print 'Predicted failures:', count_predicted_failures
print 'Total HDD count:', len(SMART_dict)
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



#REPLACE: actual values
#actual = [1,1,1,0,0,0] 
# REPLACE: actual predictions 
#predictions = [0.9,0.9,0.9,0.1,0.1,0.1]

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
