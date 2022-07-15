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
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel,Matern
from sklearn.gaussian_process.kernels import ConstantKernel as C

#Reading the dataset
df = pd.read_csv('Prior.csv',delimiter=',')
df2 = pd.read_csv('CompleteData.csv',delimiter=',')
df3= pd.read_csv('TestData.csv')

#Reading the data
fu = df.iloc[:,0].values
ep = df.iloc[:,1].values
si = df.iloc[:,2].values
y = df.iloc[:,3].values

#from complete-original dataset
fu2 = df2.iloc[:,0].values
ep2 = df2.iloc[:,1].values
si2 = df2.iloc[:,2].values
y2 = df2.iloc[:,3].values

#from testing dataset
fug_test= df3.iloc[:,0].values
fug_test= fug_test/1e5   #Pa to Bar
eps_test= df3.iloc[:,1].values
sig_test= df3.iloc[:,2].values

#Replacing y if some y value in zero
for i in range(len(y)):
        if (y[i] == 0):
                y[i] = 0.0001

#For y2
for i in range(len(y2)):
        if (y2[i] == 0):
                y2[i] = 0.0001

#Transforming 1D arrays to 2D
fu = np.atleast_2d(fu).flatten().reshape(-1,1)
ep = np.atleast_2d(ep).flatten().reshape(-1,1)
si = np.atleast_2d(si).flatten().reshape(-1,1)
y = np.atleast_2d(y).flatten()

fug_test= np.atleast_2d(fug_test).flatten().reshape(-1,1)
eps_test = np.atleast_2d(eps_test).flatten().reshape(-1,1)
sig_test = np.atleast_2d(sig_test).flatten().reshape(-1,1)

fu_true = fu
ep_true = ep
si_true = si
y_actual = y

#converting P to bars
fu = fu/(1.0e5)

#Taking logbase 10 of the input vector
fu = np.log10(fu)
ep = np.log10(ep)
si = np.log10(si)
y = np.log10(y)

#print(len(x),len(y))
#Taking the log of X_test
fug_test = np.log10(fug_test)
eps_test = np.log10(eps_test)
sig_test= np.log10(sig_test)

#Extracting the mean and std. dev for P_test
fu_m = np.mean(fug_test)
fu_std = np.std(fug_test,ddof=1)

#Extracting the mean and std. dev for T_test
ep_m = np.mean(eps_test)
ep_std = np.std(eps_test,ddof=1)

#Extracting the mean and std. dev for sig_test
si_m= np.mean(sig_test)
si_std= np.std(sig_test,ddof=1)

#Standardising p,t and y in log-space
fu_s = (fu - fu_m)/fu_std
ep_s = (ep - ep_m)/ep_std
si_s= (si - si_m)/si_std

#Standardising X_test in log-space
fug_test = (fug_test - fu_m)/fu_std
eps_test = (eps_test - ep_m)/ep_std
sig_test= (sig_test - si_m)/si_std

#Initializing scaled down training and prediction set
x_s= np.vstack((fu_s.flatten(), ep_s.flatten(),si_s.flatten())).T
X_test = np.vstack((fug_test.flatten(), eps_test.flatten(), sig_test.flatten())).T


#Building the GP regresson
# Instantiate a Gaussian Process model
# kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(10, (1e-2, 1e2))
# kernel = C(1.0, (1e-3, 1e3)) *RationalQuadratic(length_scale=50, alpha=0.5,length_scale_bounds=(1e-13,1e13),alpha_bounds=(1e-13,1e13)) + RationalQuadratic(length_scale=50, alpha=0.5,length_scale_bounds=(1e-13,1e13),alpha_bounds=(1e-13,1e13))
kernel= 2*RBF(length_scale=50,length_scale_bounds=(1e-8,1e8))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, normalize_y=True)

#Fitting our normalized data to the GP model
gp.fit(x_s,y.T)

# Make the prediction on the test data (ask for MSE as well)
y_pred, sigma = gp.predict(X_test, return_std=True)

rel_error = np.zeros(len(sigma))
#finding the relative error:
for i in range(len(sigma)):
    rel_error[i] = abs(sigma[i]/abs(y_pred[i]))

#define the limit for uncertainty
lim = 0.08
Max = np.amax(rel_error)
index = np.argmax(rel_error)

#transforming the index to original pressure point

X_test[:,0] = (X_test[:,0]*fu_std) + fu_m
X_test[:,0] = 10**(X_test[:,0])
X_test[:,0] = 1e5*(X_test[:,0])
X_test[:,0] = np.round(X_test[:,0],1)

X_test[:,1] = (X_test[:,1]*ep_std) + ep_m
X_test[:,1] = 10**(X_test[:,1])
X_test[:,1] = np.round(X_test[:,1],2)

X_test[:,2] = (X_test[:,2]*si_std) + si_m
X_test[:,2] = 10**(X_test[:,2])
X_test[:,2] = np.round(X_test[:,2],3)

xx= pd.DataFrame(X_test,columns=['fugacity','epsilon','sigma'])
xx.to_csv('Xtest.csv', index=False)
#checking the whether the maximum uncertainty is less than out desired limit
if (Max >= lim):
        Data = str(X_test[index])
        Data = Data.replace("[","")
        Data = Data.replace("]","")
        print(X_test[index,0],X_test[index,1], X_test[index,2] )
        print("NOT_DONE ")
        print(rel_error[index])
else:
        Data = str(X_test[index])
        Data = Data.replace("[","")
        Data = Data.replace("]","")
        print(X_test[index,0],X_test[index,1], X_test[index,2])
        print("DONE")
        print("Final Maximum Error=", rel_error[index])

y_pred = 10**y_pred
pred= y_pred
#print(np.shape(X_test),np.shape(y_actual))

rel_error = 100*(rel_error)

#Finding relative root mean square error
rrmse = 0
for i in range(len(rel_error)):
        rrmse = rrmse + (((y_pred[i] - y2[i])**2)/y2[i]**2)
        X= (((y_pred[i] - y2[i])**2)/y2[i]**2)
        #print(X,y_pred[i],y2[i])
rrmse = 100*(np.sqrt(rrmse))/len(rel_error)

#defining relative true error
rel_t = np.zeros(len(X_test))
for i in range(len(rel_t)):
        rel_t[i] = ((y_pred[i] - y2[i])/y2[i])
        #print(rel_t[i])

### This piece of code block below is simply for testing purposes, i.e. comparing the test set and grouth truth results

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


# Printing the final predicted data
a=pd.DataFrame(pred,columns=['Predicted'])
a.to_csv('pred.csv',index=False)
