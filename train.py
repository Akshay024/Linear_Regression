import csv 
import numpy as np
import pandas as pd

def import_data():
    train_X=np.genfromtxt("C:/Users/Akshay/Downloads/linear_regression/train_X_lr.csv",delimiter=',', dtype=np.float64, skip_header=1)
    train_Y=np.genfromtxt("C:/Users/Akshay/Downloads/linear_regression/train_Y_lr.csv",delimiter=',', dtype=np.float64)
    return train_X,train_Y

def compute_gradient_of_cost_func(X,Y,W):
    Y_pred=np.dot(X,W)
    diff=Y_pred-Y
    dW=(1/len(X))* np.dot(diff.T,X)
    dW=dW.T

    return dW

def compute_cost(X, Y, W):
    y_pred=np.dot(X,W)
    diff=y_pred-Y
    mse=np.sum(np.square(diff))
    cost=mse/(2*len(X))
    return cost

def optimize_weights_using_gradient_descent(X,Y,W,num_iteration, learning_rate):
    for i in range(num_iteration):
        dW=compute_gradient_of_cost_func(X,Y,W)
        W=W-(learning_rate*dW)
        cost=compute_cost(X,Y,W)
        print(i,cost)

    return W

def train_model(X,Y):
    X=np.insert(X,0,1,axis=1)
    Y=Y.reshape(len(X),1)
    W=np.zeros((X.shape[1],1))
    W=optimize_weights_using_gradient_descent(X,Y,W,10,0.0001)

    return W
    
def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w') as weights_file:
        wr= csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()
        

if __name__ =="__main__":
    X,Y=import_data()
    weights=train_model(X,Y)
    save_model(weights, "WEIGHTS_FILE.csv")
    
