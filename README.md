# ML_final_project
This file is an instruction set explaining how to run the code files submitted 

To run any of our code files, unzip the code.zip and data.zip files in a shared directory.  Due to size limitations
the data provided is not correctly formatted to be read in by the models and the python code must be modified to read 
in the small subset of data submitted. 

import_data.py 


Baseline.py
  - code to implement baseline prediction
  - Baseline predicts failure if any values in its selected SMART attributes is greater than 0 and predicts no
    failure otherwise. 
  - Inputs: test data set 
  - Outputs: TP, FP, FN, TN counts as well as calculated rates, precision, accuracy and recall 
  
  command to run: python Baseline.py 
  
LogisticRegression.py

NaiveBayes.py

RandomForest.py
  - code used to implement RandomForest on test and training data
  - Inputs: training data sets and test/ validation data set 
  - Outputs: TP, TN, FP, FN and a ROC curve 
  
  command to run: 
