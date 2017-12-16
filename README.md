# ML_final_project
This file is an instruction set explaining how to run the code files submitted 

To run any of our code files, unzip the code.zip and data.zip files in a shared directory.  Due to size limitations
the data provided is not correctly formatted to be read in by the models and the python code must be modified to read 
in the small subset of data submitted. 

import_data.py
  - code that preprocesses data and prepares it for training / testing
  - filters out features such that only the features we want are returned for training / testing


Baseline.py
  - code to implement baseline prediction
  - Baseline predicts failure if any values in its selected SMART attributes is greater than 0 and predicts no
    failure otherwise. 
  - Inputs: test data set 
  - Outputs: TP, FP, FN, TN counts as well as calculated rates, precision, accuracy and recall 
  
  command to run: python Baseline.py 
  
LogisticRegression.py
  - code that implements logistic regression
  - uses LogisticRegression module from sci-py for training and testing
  - feature values are preprocessed where value is 1 if raw value > 0 and 0 otherwise
  - Inputs (inside code): training data set and test data set
  - Outputs: TP, FP, FN, TN counts as well as calculated rates, precision, accuracy, recall, and ROC curves

NaiveBayes.py
  - code used to implement NaiveBayes on test and training data
  - Uses MultinomialNaiveBayes module from sci-py for training and testing
  - Inputs: test data set
  - Outputs: TP, FP, FN, TN counts as well as calculated rates, precision, accuracy and recall
  
  command to run: python Naive_Bayes.py
  
  (Note the script is in the branch naive-bayes)
  
RandomForest.py
  - code used to implement RandomForest on test and training data
  - Inputs: training data sets and test/ validation data set 
  - Outputs: TP, TN, FP, FN and a ROC curve 
  
  command to run (for training and testing):
  py -2 ./<model_name>.py
  
  Note: the command assumes that training and testing files are found in "../test_10/" for training data and "../test_final/" for test data.
