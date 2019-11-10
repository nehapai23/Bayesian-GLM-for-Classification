#!/usr/local/bin/python3
import sys
import numpy as np 
import pandas as pd
import random
import math 
import matplotlib.pyplot as plt  
import pprint
import timeit

#Function to read dataset based on name and randomly split into training and testing set
def read_csv(dataset_name):
    features = pd.read_csv("pp3data/"+dataset_name+".csv", header = None).values
    labels = pd.read_csv("pp3data/labels-"+dataset_name+".csv", header = None).values
    #irlstest is sample dataset to test w, we do not split it
    if dataset_name == "irlstest":
        return features, labels, features, labels
    else:
        dataset = np.column_stack((features,labels))
        #np.random.shuffle(dataset)
        #training set is 2/3rd of dataset and test set is remaining
        train_l = int(2/3*len(dataset))
        train, trainL, test, testL = dataset[:train_l,:-1], dataset[:train_l,-1:], dataset[train_l:,:-1], dataset[train_l:,-1:]
        return train, trainL, test, testL

#Function to compute 1st and 2nd derivative for Logistic Regression
def compute_R_d_Logistic(a, t): 
    y = sigmoid(-a)           
    r = y * (1 - y)
    #First Derivative term
    d = t-y
    #Second Derivative term R is diagonal matrix of y(1-y)
    R = np.diag(r.ravel())
    return R, d

#Function to compute 1st and 2nd derivative for Poisson Regression
def compute_R_d_Poisson(a, t): 
    y = np.array([[math.exp(ai)] for ai in a])          
    r = y
    #First Derivative term
    d = t-y
    #Second Derivative term R is diagonal matrix of y
    R = np.diag(r.ravel())
    return R, d

#Function to compute 1st and 2nd derivative for Ordinal Regression
def compute_R_d_Ordinal(a, t): 
    phiJ = [-math.inf,-2,-1,0,1,math.inf]
    s = 1
    d = []
    r = []
    for i,ai in enumerate(a):
        ti = int(t[i][0])
        yiti = yij(ai,phiJ[ti],s)
        yiti_1 = yij(ai,phiJ[ti-1],s)
        d.append(yiti + yiti_1 - 1)
        r.append(s*s*(yiti*(1-yiti)+yiti_1*(yiti_1)))
    #print(d)
    #print(r)
    ri = np.array(r)
    R = np.diag(ri.ravel())
    return R, d

#Prediction function for Logistic regression
def prediction_Logistic(a):
    y = sigmoid(-a) 
    #Predict True label for values >=0.5
    y = [int(val>=0.5) for val in y] 
    return y  

#Prediction function for Poisson regression
def prediction_Poisson(a):
    y = [math.exp(ai) for ai in a]
    #Use floor function to predict 
    t = [math.floor(yi) for yi in y]
    return t 

#Prediction function for Ordinal regression
def prediction_Ordinal(a):
    s = 1
    phiJ = [-math.inf,-2,-1,0,1,math.inf]
    t = []
    #Compute values for all ordinals J = 1,2,3,4,5 and choose max
    for ai in a:
        pj = []
        for j in range(1,6):
            yj = yij(ai,phiJ[j],s)
            yj_1 = yij(ai,phiJ[j-1],s)
            pj.append(yj - yj_1)
        t.append(pj.index(max(pj))+1)
    return t 

#Function to plot the error rate as a function of training set sizes
def plot_summary(data,sizes,model,dataset_name,alpha):
    errors = [d.get("Mean") for d in data]
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    plt.rc('font', **font)
    std = [d.get("STD") for d in data]
    plt.gcf().clear()
    plt.figure(figsize=(25,25),dpi=90)
    plt.errorbar(sizes,errors,yerr=std,ecolor='r', color = 'b', capsize=25, label = "GLM Model : "+model)
    plt.xlabel("Training Sizes")
    plt.ylabel("Error Rate")
    plt.grid("on")
    plt.title("Dataset: " +dataset_name+" | Alpha: "+str(alpha))
    plt.legend(loc="best")
    plt.savefig(dataset_name + "_"+ model + '.png')
    plt.show()

#Function implement the common GLM function
def GLM_variant(model, dataset_name):
    #Read dataset and split to Train and Test sets
    train, trainL, test, testL = read_csv(dataset_name)
    #Set training set sizes as 0.1, 0.2, 0.3... 1 
    training_set_sizes = [1] if dataset_name == "irlstest" else [int(i/10*len(train)) for i in range(1, 11, 1)]
    summary = []
    for size in training_set_sizes:
        trials = 30 if dataset_name != "irlstest" else 1
        #trials =1
        error_predictions = []
        iterations = []
        time = []
        #Repeat for 30 trials
        for trial in range(0,trials):
            if dataset_name == "irlstest":
                train_sub, trainL_sub = train, trainL
            else:    
                #Shuffle training data
                #train_sub, trainL_sub = zip(*random.sample(list(zip(train, trainL)),size))
                train_sub, trainL_sub = train[:size], train[:size]
            t = trainL_sub
            N = len(train_sub)
            w0 = np.array([[1]] * N)
            #Append data matrix with ones
            phi = np.concatenate((w0, train_sub), axis=1)

            M = len(phi[0])
            
            #Set parameter value
            alpha = 10

            I = np.eye(M)
            #Newton Raphson starts with w0 = vector of zeroes
            w = np.array([[0]] * M)
            convergence_test = 1
            itr = 1
            start = timeit.default_timer()
            #Repeat Newton Raphson update formula until convergence or 100 iterations
            while itr < 100 and convergence_test > 10 ** -3:
                w_old = w
                a = np.matmul(phi,w_old)
                #Compute first and second Derivatives based on Model
                if model == "logistic": 
                    R, d = compute_R_d_Logistic(a, t)
                elif model == "poisson":
                    R, d = compute_R_d_Poisson(a, t)
                elif model == "ordinal":
                    R, d = compute_R_d_Ordinal(a, t)

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
            #print(w)  
            stop = timeit.default_timer()      
            iterations.append(itr)
            time.append(stop-start)
            #prediction
            Ntest = len(test)
            w0 = np.array([[1]] * Ntest)
            #Use test set for prediction
            phi = np.concatenate((w0, test), axis=1)
            #We predict using WMap calculated earlier using Newton Raphson
            a = np.matmul(phi,w)
            if model == "logistic": 
                y = prediction_Logistic(a)
                error_predictions.append(((y-testL.flatten()) != 0).sum())
            if model == "poisson":
                y = prediction_Poisson(a)
                error_predictions.append((abs(y-testL.flatten()) != 0).sum())
            if model == "ordinal":
                y = prediction_Ordinal(a)
                error_predictions.append((abs(y-testL.flatten()) != 0).sum())
        
        print(size,"Done")
        summary_data = {}
        summary_data['Model'] = model
        summary_data['Run Time'] = np.mean(time)
        summary_data['Dataset Size'] = size
        summary_data['Mean'] = np.mean(np.array(error_predictions)/Ntest)
        summary_data['STD'] = np.std(np.array(error_predictions)/Ntest)
        summary_data['Iterations'] = np.mean(iterations)
        summary.append(summary_data)
    pprint.pprint(summary)
    filename = 'Output Summary '+model+ ' ' + dataset_name + '.txt'
    #Write Summary to file
    with open(filename, 'wt') as out:
        pprint.pprint(summary, stream=out)
    #Plot graph
    plot_summary(summary,training_set_sizes,model, dataset_name, alpha)

#Utility function to calculate Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(x))


#Utility function to calculate sigmoid based on S and Phi parameters for Ordinal
def yij(a,phij,s):
    x = np.array(s*(phij-a))
    if x >= 0:
        z = np.exp(-1*x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

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
    GLM_variant(model, dataset_name)

    print("\n\n..Done!")


