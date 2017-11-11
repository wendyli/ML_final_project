import csv
import json


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

days = [31, 28, 31]

for m in xrange(1,3):

	day = days[m-1]	
	for i in range(1,day):

		name_i = '0' + str(i) if i < 10 else str(i)
		file_data_name = "data_Q1_2017/2017-0{0}-{1}.csv".format(m, name_i)

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

				#print 'status_predicted:', attribute_dict['status_predicted']
				#print 'predicted_fail_date:', attribute_dict['predicted_fail_date']
				#print ''

				SMART_dict[serial_number] = attribute_dict


		# # go through data list and append hard disk drive values
		# for i in range(0, len(actual_data_list)): # iterate through all data values
		# 	# get row of each individual hard drive's data list
		# 	hard_drive_data = actual_data_list[i][0].split(',') # individual hard drive's data
			
		# 	for j, key in enumerate(attributes_list): # get the specific attribute 
		# 		if (key in important_attributes):
		# 			SMART_dict[key].append(hard_drive_data[j]) # append data of individual hard drive to disk


		#

count_actual_failures = 0
count_predicted_failures = 0

false_pos = 0
false_neg = 0

num_examples = len(SMART_dict)

for key in SMART_dict:
	#print "Current hd: " + key
	curr_attribute_dict = SMART_dict[key]
	count_actual_failures += curr_attribute_dict['actual_status']
	count_predicted_failures += curr_attribute_dict['status_predicted']

	if curr_attribute_dict['status_predicted'] == 1 and curr_attribute_dict['actual_status'] == 0:
		false_pos += 1
	elif curr_attribute_dict['status_predicted'] == 0 and curr_attribute_dict['actual_status'] == 1:
		false_neg += 1

	
print num_examples
print 'Actual failures:', count_actual_failures
print 'Predicted failures:', count_predicted_failures
print "False positive: {0}".format(false_pos)
print "False negative: {0}".format(false_neg)
