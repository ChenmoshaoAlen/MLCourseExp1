import numpy as np
# import pandas as pd
import sklearn.datasets as sd
import sklearn.model_selection as sms
# import matplotlib.pyplot as plt
# import math
# import random

# exp1
X, y = sd.load_svmlight_file('housing_scale.txt',n_features = 13)
X_train, X_valid, y_train, y_valid = sms.train_test_split(X, y,test_size=0.1,shuffle=True)

X_train = X_train.toarray()
X_valid = X_valid.toarray()
y_train = y_train.reshape(len(y_train),1)
y_valid = y_valid.reshape(len(y_valid),1)#转化为1列

def compute_loss(X, y, theta):
    '''均方误差损失'''
    hx = X.dot(theta)#w点乘X
    error = np.power((hx - y), 2).mean()
    return error

def normal_equation(X, y):
    return (np.linalg.inv(X.T.dot(X))).dot(X.T).dot(y)

theta = normal_equation(X_train, y_train)

loss_train = compute_loss(X_train, y_train, theta)
loss_valid = compute_loss(X_valid, y_valid, theta)

print(loss_train)
print(loss_valid)

# exp2