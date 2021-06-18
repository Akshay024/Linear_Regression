from os import path
import numpy as np
import csv


def check_file_exits():
    if not path.exists("predicted_test_Y_lr.csv"):
        raise Exception("Couldn't find '" + "predicted_test_Y_lr.csv"  +"' file")

def check_format(test_X_file_path):
    pred_Y = []
    with open("predicted_test_Y_lr.csv", 'r') as file:
        reader = csv.reader(file)
        pred_Y = list(reader)
    pred_Y = np.array(pred_Y)

    test_X = np.genfromtxt(test_X_file_path, delimiter=',', \
        dtype=np.float64, skip_header=1)

    if pred_Y.shape != (len(test_X), 1):
        raise Exception("Output format is not proper")

def check_mse():
    pred_Y = np.genfromtxt("predicted_test_Y_lr.csv", delimiter=',', dtype=np.float64)
    actual_Y = np.genfromtxt("train_Y_lr.csv", delimiter=',', dtype=np.float64)

    difference_array = np.subtract(actual_Y, pred_Y)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    print("MSE", mse)
    return mse

def validate(test_X_file_path):

    check_file_exits()
    check_format(test_X_file_path)
    check_mse()
