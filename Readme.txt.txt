#####################################################################
The main code is organized in pp3.py file. 
The Datasets are assumed to be in folder - 'pp3data/'
The code can be executed as ->

python .\pp3.py <dataset_name> <Model_name>

Sample dataset names = ["A","usps","AO","AP"]
Sample model names = ["logistic","poisson","ordinal"]

The code plots the error rate as a function of training set sizes
The code also generates a output summary with following fields:
1)'Dataset Size': Size of training data set
2)'Iterations': Mean number of 30 trials
3)'Mean': Average error for 30 trials
4)'Model': Model name
5)'Run Time': Mean Running time for Wmap computation
6)'STD': Mean Standard Deviation

The filename is ‘Output Summary <model name> <dataset name>.txt’ 

#####################################################################

For Model selection, the code is organized in model_selection.py
The code can be executed as ->

python .\model_selection.py <dataset_name> <Model_name>

Sample dataset names = ["A","usps","AO","AP"]
Sample model names = ["logistic","poisson","ordinal"]

#####################################################################
