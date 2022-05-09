#!/usr/bin/env python
# coding: utf-8

#importing important libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math

#importing the ML libraries
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, Matern
from sklearn.gaussian_process.kernels import ConstantKernel as C

#Reading the dataset
df = pd.read_csv('Prior.csv',delimiter=',')
df2 = pd.read_csv('CompleteData.csv',delimiter=',')
#print(df.head())

#sorting out the data
#df = df.sort_values(by=['Pressure'], ascending='True')
#print(df.head(n=40))

#Substitute a data row with some non-zero value if methane uptake for that row zero, this is to avoid log(0) = NaN error
#df['Uptake_for_Cu-BTC-CH4(300K)'].replace(to_replace=0, value=0.00001)
#print(df.head(n=40))

#print(df)
#Unseen array
X_test_1= np.linspace(1e-5,1e-4,9)
X_test_2= np.linspace(1.1e-4,1e-3,9)
X_test_3= np.linspace(1.1e-3,1e-2,9)
X_test_4= np.linspace(1.1e-2,1e-1,9)
X_test_5= np.linspace(1.1e-1,1,10)
X_test=np.concatenate([X_test_1,X_test_2,X_test_3,X_test_4,X_test_5]).flatten().reshape(-1,1)

#Reading the data
x = df.iloc[:,0].values
y = df.iloc[:,1].values
#Taking the error data as well
e = df.iloc[:,2].values

#from complete-original dataset
x2 = df2.iloc[:,0].values
y2 = df2.iloc[:,1].values

#Replacing y if some y value in zero
for i in range(len(y)):
  if (y[i] == 0):
      y[i] = 0.0001

#For y2
for i in range(len(y2)):
  if (y2[i] == 0):
      y2[i] = 0.0001

#Transforming 1D arrays to 2D
x = np.atleast_2d(x).flatten().reshape(-1,1)
y = np.atleast_2d(y).flatten().reshape(-1,1)

x_true = x
y_actual = y
#before the transition
#print(x,y)

#converting P to bars
x = x/(1.0e5)

#Taking logbase 10 of the input vector
x = np.log10(x)
y = np.log10(y)

#print(len(x),len(y))
#Taking the log of X_test
X_test = np.log10(X_test)

#Extracting the mean and std. dev for X_test
x_m = np.mean(X_test)
x_std = np.std(X_test,ddof=1)

#Standardising x and y in log-space
x_s = (x - x_m)/x_std

#Standardising X_test in log-space
X_test = (X_test - x_m)/x_std

#print(x_s,y)
#Building the GP regresson 
# Instantiate a Gaussian Process model
#kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(10, (1e-2, 1e2))
kernel = RationalQuadratic(length_scale=50, alpha=0.5,length_scale_bounds=(1e-8,1e8),alpha_bounds=(1e-8,1e8))
#kernel= Matern(length_scale=1, nu=0.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, normalize_y=True)

#Fitting our normalized data to the GP model
gp.fit(x_s,y)

#print(X_test)
# Make the prediction on the test data (ask for MSE as well)
y_pred, sigma = gp.predict(X_test, return_std=True)
#print(y_pred,sigma)
rel_error = np.zeros(len(sigma))

#finding the relative error—
for i in range(len(sigma)):
    rel_error[i] = abs(sigma[i])
# print(rel_error)
#define the limit for uncertainty
lim = 0.02
Max = np.amax(rel_error)
index = np.argmax(rel_error)

#transforming the index to original pressure point
X_test = (X_test*x_std) + x_m
X_test = 10**(X_test)
X_test = 1e5*(X_test)
#print(X_test,10**(y_pred),rel_error)

#checking the whether the maximum uncertainty is less than out desired limit
if (Max >= lim):
  Data = str(X_test[index])
  Data = Data.replace("[","")
  Data = Data.replace("]","")
  print(Data)
  print("NOT_DONE ")
  print(rel_error[index])
else:
  Data = str(X_test[index])
  Data = Data.replace("[","")
  Data = Data.replace("]","")
  print(Data)
  print("DONE")
  print("Final Maximum Error=", rel_error[index])

y_pred = 10**y_pred
pred= y_pred
#print(np.shape(X_test),np.shape(y_actual))
#print(X_test,y_pred)

rel_error = 100*(rel_error)

#defining relative root mean square error
#defining relative true error
rel_t = np.zeros(len(X_test))
for i in range(len(rel_t)):
  rel_t[i] = ((y_pred[i] - y2[i])/y2[i])
  #print(X_test[i],x2[i],y_pred[i],y2[i])
#   print(rel_t[i])

#Finding relative root mean square error
rrmse = 0
for i in range(len(rel_error)):
  rrmse = rrmse + (((y_pred[i] - y2[i])**2)/y2[i]**2)
  X= (((y_pred[i] - y2[i])**2)/y2[i]**2)
  #print(X_test[i],p2[i],t2[i],y_pred[i],y2[i])
rrmse = 100*(np.sqrt(rrmse))/len(rel_error)
# print(rrmse)

#converting the true rel error in percentage
rel_t = 100*(abs(rel_t))

#finding the mean of relative error
rel_m = np.mean(rel_error)

#printing mean of rel error and rrmse for each iteration in a separate mean.csv file
os.system("echo -n "+str(rel_m)+","+str(rrmse)+" >> mean.csv")

##printing rel error and true rel. error in a error file
for i in range(len(rel_error)):
    #rounding off the error to 3 digits after decimals
    rel_t[i] = round(rel_t[i],3)
    rel_error[i] = round(rel_error[i],3)
    #printing them in the .csv files for error
    os.system("echo -n "+str(rel_error[i])+" >> rel.csv")
    os.system("echo -n "+str(rel_t[i])+" >> rel_true.csv")
    if ( i != (len(rel_error) - 1)): 
        os.system("echo -n "+","+" >> rel.csv")
        os.system("echo -n "+","+" >> rel_true.csv")

#Printing the final predicted data
a=pd.DataFrame(pred,columns=['Predicted'])
a.to_csv('pred.csv',index=False)

#Printing the X_test
b=pd.DataFrame(X_test, columns=['X Test'])
b.to_csv('X_test.csv', index=False)
