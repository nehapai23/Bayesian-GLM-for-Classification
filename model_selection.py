#!/usr/local/bin/python3
import sys
import numpy as np 
import pandas as pd
import random
import math 
import matplotlib.pyplot as plt  
import pprint
import timeit
from numpy import linalg as LA
import pp3

#Function to perform cross validation for model selection. Based on my code in Assignment 2
def model_selection_using_cross_validation(train, trainR, test, testR, dataset_name, model):
    len_train = np.shape(train)[0]
    step = len_train//10 
    error_for_params = []  
    #Running cross validation with below values of parameters
    test_params = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] + [i for i in range(1,101)]  
    for l in test_params:
        error_predictions = []
        #Steps = 10 to perform 10 fold cross validation
        for i in range(0,len_train, step):
            #Training set will be all portions except i-th portion
            current_train = np.delete(train,slice(i,i+step),0)
            current_trainR = np.delete(trainR,slice(i,i+step),0)
            w = calculate_w(l,current_train,current_trainR, model)
            
            test = train[i:i+step]
            testL = trainR[i:i+step]
            Ntest = len(test)
            w0 = np.array([[1]] * Ntest)
            #Use remaining part of data set for prediction
            phi = np.concatenate((w0, test), axis=1)
            a = np.matmul(phi,w)
            #We predict using WMap calculated earlier using Newton Raphson
            if model == "logistic": 
                y = pp3.prediction_Logistic(a)
                error_predictions.append(((y-testL.flatten()) != 0).sum())
            if model == "poisson":
                y = pp3.prediction_Poisson(a)
                error_predictions.append((abs(y-testL.flatten()) != 0).sum())
            if model == "ordinal":
                y = pp3.prediction_Ordinal(a)
                error_predictions.append((abs(y-testL.flatten()) != 0).sum())
        error_for_params.append(avg(error_predictions))
    
    print("Dataset: ", dataset_name)
    print("--MODEL SELECTION USING CROSS VALIDATION--")
    print("Parameter: ", str(test_params[error_for_params.index(min(error_for_params))]))


#Using Same function to calculate w
#Didn't get enough time to generalize function wriiten in pp3 to return only w for cross validation. 
#But I am calling functions written in pp3 file
def calculate_w(l, train_sub, trainL_sub, model):
    t = trainL_sub
    N = len(train_sub)
    w0 = np.array([[1]] * N)
    #Append data matrix with ones
    phi = np.concatenate((w0, train_sub), axis=1)

    M = len(phi[0])
    
    #Set parameter value
    alpha = l

    I = np.eye(M)
    #Newton Raphson starts with w0 = vector of zeroes
    w = np.array([[0]] * M)
    convergence_test = 1
    itr = 1
    #Repeat Newton Raphson update formula until convergence or 100 iterations
    while itr < 100 and convergence_test > 10 ** -3:
        w_old = w
        a = np.matmul(phi,w_old)
        #Compute first and second Derivatives based on Model
        if model == "logistic": 
            R, d = pp3.compute_R_d_Logistic(a, t)
        elif model == "poisson":
            R, d = pp3.compute_R_d_Poisson(a, t)
        elif model == "ordinal":
            R, d = pp3.compute_R_d_Ordinal(a, t)

        #First derivative
        g = np.matmul(np.transpose(phi),d) - (alpha * w)
        #Hessian matrix of second derivatives
        H = -(alpha * I) - np.matmul(np.transpose(phi),np.matmul(R,phi))
        #Newton Raphson update formula for GLM 
        #W_old = W_new - inverse(H)*g
        if np.linalg.det(H) != 0:
            w_new = w_old - np.matmul(np.linalg.inv(H),g)
        #Test convergence
        if  np.linalg.norm(w_old) != 0:
            convergence_test = np.linalg.norm(w_new - w_old) / np.linalg.norm(w_old)
        
        w = w_new
        itr += 1
    return w

#Find average of list
def avg(lst):
    return sum(lst)/len(lst)


#Code Execution starts here!
if __name__ == "__main__":
    #Sample dataset names = ["A","usps","AO","AP"]
    #Sample model names = ["logistic","poisson","ordinal"]
    if(len(sys.argv) != 3):
        raise Exception('Error: expected 2 command line arguments!')

    #Code expects dataset name and model name as command line argument    
    dataset_name = sys.argv[1]
    model = sys.argv[2]
    #Common function to generate GLM model, predict and evaluate

    train, trainL, test, testL = pp3.read_csv(dataset_name)
    model_selection_using_cross_validation(train, trainL, test, testL, dataset_name, model)
    print("..Done!")


