# Sina Mp. Saravani
# Oct 2019
# Advanced Big Data Analytics
import numpy as np
import sys
import matplotlib.pyplot as plt
from texttable import Texttable

# read call arguments to a variable
args = sys.argv

# read .csv file of training and test set
trainset = np.loadtxt(open(args[1], "r"), delimiter=",")
testset = np.loadtxt(open(args[2], "r"), delimiter=",")

# preparing appropriate test data matrices
matXtest = testset[:, [i for i in range(testset.shape[1]-1)]]
matInttest = [1 for i in range(testset.shape[0])]
matInttest = np.reshape(matInttest, [testset.shape[0], 1])
matXtest = np.concatenate((matInttest, matXtest), axis=1)
matYtest = testset[:, testset.shape[1]-1]

# preparing appropriate train data matrices
matX = trainset[:, [i for i in range(trainset.shape[1]-1)]]
matInt = [1 for i in range(trainset.shape[0])]
matInt = np.reshape(matInt, [trainset.shape[0], 1])
matX = np.concatenate((matInt, matX), axis=1)
matY = trainset[:, trainset.shape[1]-1]

# going trough the matrix calculation process for acquiring coefficients (b)
matXtrans = np.transpose(matX)
matXtransX = np.matmul(matXtrans, matX)
try:
    matXtransXinv = np.linalg.inv(matXtransX)
except np.linalg.LinAlgError:
    # Not invertible. Skip this one.
    print("The X'X matrix is not invertible.")
    pass
else:
    # Invertible, continue the program
    tmp = np.matmul(matXtransXinv, matXtrans)
    coefs = np.matmul(tmp, matY)
    coefstrans = np.reshape(coefs, [matX.shape[1], 1])

    # print the coefficients table:
    t = Texttable()
    t.set_max_width(0)
    headers = ['c_' + str(i) for i in range(1, matXtest.shape[1])]
    headers = ['Intercept'] + headers
    t.add_rows([headers, coefs])
    print(t.draw())

    # predict the output for test data
    res = np.matmul(matXtest, coefstrans)

    # calculate the RMSE and print it
    sum = 0
    for i in range(matXtest.shape[0]):
        sum += (matYtest[i] - res[i])**2
        # sum += np.abs(matYtest[i] - res[i])
    mean = sum/matXtest.shape[0]
    rmse = mean ** 0.5
    print("RMSE ERROR:")
    print(rmse[0])

    # draw the figure for predictions vs. ground truth
    plt.scatter(res, matYtest)
    plt.xlabel("Prediction")
    plt.ylabel("Ground truth")
    x1, y1 = [0, np.maximum(np.max(res), np.max(matYtest))], [0, np.maximum(np.max(res), np.max(matYtest))]
    plt.plot(x1, y1)
    plt.show()
    exit(0)
