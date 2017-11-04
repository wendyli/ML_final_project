import csv

file_data_name = "data_Q1_2017/2017-01-01.csv"


# Returns tuple consisting of the following:
# - the set of attribute names
# - a map from a HDD serial number to its data vector, where each data vector
#   is a map from an attribute name to its corresponding integer value
def extractData(filename):
    with open(filename, 'r') as file_data:
        reader = csv.reader(file_data, delimiter=',')
        data_list = list(reader)

        actual_data_list = data_list[1:]
        attributes_list = data_list[0]

        serial_num_to_data_vector = {}

        num_features = len(attributes_list)    
        num_examples = len(actual_data_list) 
        for i in range(0, num_examples):
            ith_raw_data = actual_data_list[i]

            ith_data_vector = {}
            
            # First 3 are data, serial number, and model, which are irrelevant
            for j in range(3, num_features):

                if (ith_raw_data[j] != ''):
                    ith_data_vector[attributes_list[j]] = int(ith_raw_data[j])

            serial_num_to_data_vector[ith_raw_data[1]] = ith_data_vector
        
        return (set(attributes_list), serial_num_to_data_vector)
