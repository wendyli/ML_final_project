import csv

file_data_name = "data_Q1_2017/2017-01-01.csv"

with open(file_data_name, 'r') as file_data:
	reader = csv.reader(file_data, delimiter='\t')
	data_list = list(reader)

	actual_data_list = data_list[1:]
	attributes_list = data_list[0][0].split(',')

	print(attributes_list)


	hd_to_feature = {}

	num_features = len(attributes_list)    
	num_examples = len(actual_data_list) 
	for i in range(0, num_examples):
		hard_drive_data = actual_data_list[i][0].split(',')

		feature_to_data = {}
		for j in range(3, num_features):

			if (hard_drive_data[j] != ''):
				feature_to_data[attributes_list[j]] = hard_drive_data[j]

		hd_to_feature[hard_drive_data[1]] = feature_to_data 

		print hd_to_feature