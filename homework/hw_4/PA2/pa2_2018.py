# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:13:23 2018

@author: xietianwen
"""

import numpy as np
from numpy import linalg

# load vowel data stored in npy
'''
NOTICE:
labels of y are: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
'''
x_test = np.load('x_test.npy')
print('x_test\'s shape: {}'.format(x_test.shape))
y_test = np.load('y_test.npy')
print('y_test\'s shape: {}'.format(y_test.shape))
x_train = np.load('x_train.npy')
print('x_train\'s shape: {}'.format(x_train.shape))
y_train = np.load('y_train.npy')
print('y_train\'s shape: {}'.format(y_train.shape))

pi = 3.1415926  # value of pi

'''
x : m * n matrix
u : 1 * n vector
sigma ： n * n mtrix
result : 1 * m vector
the function accept x,u ,sigma as parameters and return corresponding probability of N(u,sigma)
you can choise use it to claculate probability if you understand what this function is doing 
your choice!
'''
def density(x,u,sigma):
    n = x.shape[1]
    buff = -0.5*((x-u).dot(linalg.inv(sigma)).dot((x-u).transpose()))
    exp = np.exp(buff)
    C = 1 / np.sqrt(np.power(2*pi,n)*linalg.det(sigma))
    result = np.diag(C*exp)
    return result


'''
class GDA
self.X : training data X
self.Y : training label Y
self.is_linear : True for LDA ,False for QDA ,default True
please make youself konw basic konwledge about python class programming

tips : function you may use
np.histogram(bins = self.n_class)
np.reshape()
np.transpose()
np.dot()
np.argmax()

'''
class GDA():
    def __init__(self, X, Y, is_linear = True):
        self.X = X
        self.Y = Y
        self.is_linear =  is_linear
        self.n_class = len(np.unique(y_train)) # number of class , 11 in this problem 
        self.n_feature = self.X.shape[1]       # feature dimention , 10 in this problem
        
        self.pro = np.zeros(self.n_class)     # variable stores the probability of each class
        self.mean = np.zeros((self.n_class,self.n_feature)) #variable store the mean of each class
        self.sigma = np.zeros((self.n_class,self.n_feature,self.n_feature)) # variable store the covariance of each class
        
        
    def calculate_pro(self):  
        #calculate the probability of each class and store them in  self.pro
        '''
        your code
        
        '''
        m = self.X.shape[0]#the size of training set
        p = np.zeros((11,1))
        for i in range(0,m):
            p[self.Y[i][0]-1][0]+=1
        self.pro = p/m
        #print(self.pro)
        
    
    def claculate_mean(self):
        #calculate the mean of each class and store them in  self.mean
        '''
        your code
        '''
        m = self.X.shape[0]#the size of training set
        means = np.zeros((11,10))
        num = [0.0 for j in range(11)]
        for i in range(0,m):
            means[self.Y[i][0]-1] += self.X[i]
            num[self.Y[i][0]-1]+=1
        for i in range(11):
            means[i] = means[i]/num[i]
        #print(num)
        self.mean = means
        #print("mean",self.mean)


        
        
            
    def claculate_sigma(self):
        #calculate the covariance of each class and store them in  self.sigma
        '''
        your code
        '''
        m = self.X.shape[0]
        sigma = np. zeros((10,10))
        # LDA
        if self.is_linear:
            for i in range(m):
                temp = np.reshape(self.X[i] - self.mean[self.Y[i][0]-1],(10,1))
                sigma +=temp.transpose() * temp
                #print( sigma)
            sigma = sigma / m
            for i in range(11):
                self.sigma[i] = sigma
        #QDA
        else:
            sigmas = np.zeros((11,10,10))
            num = [0 for i in range(11)]
            for i in range(m):
                temp = np.reshape(self.X[i] - self.mean[self.Y[i][0]-1],(10,1))
                sigmas[self.Y[i][0]-1]+= temp.transpose() * temp
                num[self.Y[i][0]-1]+=1
            for i in range(11):
                sigmas[i] /= num[i]
            self.sigma =sigmas 
        
        
        
     
    def classify(self,x_test):
        # after training , use the model to classify x_test, return y_pre
        '''
        your code
        '''
        m = x_test.shape[0]

        #I hoped to accelerate this process, but the effect may be little, because I am not very familiar with the numpy's api.
        #Use my own way to compute the probability 
        
        for i in range(11):
            sigma = self.sigma[i]
            means = np.reshape(np.repeat(self.mean[i],m,axis=0),(10,m)).transpose()#mean matrix
            ll = linalg.det(sigma)
            temp = np.mat(x_test - means)#m*n
            mn = temp*linalg.inv(sigma)#m*n
            m1 = np.zeros((m,1))
            for j in range(m):
                m1[j][0] = (mn[j]*temp[j].transpose())[0][0]
            if i == 0:
                final = np.exp(-m1*0.5)/(np.sqrt(pow(2*pi,10)*ll))
            else: 
                final = np.vstack((final,np.exp(-m1/2)/(np.sqrt(pow(2*pi,10)*ll))))
        
        final = np.reshape(final,(11,m))
        final = final.transpose()
        y_pre = np.zeros((m,1))
        for i in range(m):
            y_pre[i][0] = np.argmax(final[i])+1
        
        return y_pre

    
LDA = GDA(x_train,y_train) # generate the LDA model
LDA.calculate_pro()        # calculate parameters
LDA.claculate_mean()
LDA.claculate_sigma()
y_pre = LDA.classify(x_test) # do classify after training
LDA_acc = (y_pre == y_test).mean()
print ('accuracy of LDA is:{:.2f}'.format(LDA_acc))   
    

QDA = GDA(x_train,y_train,is_linear=False) # generate the QDA model
QDA.calculate_pro()                     # calculate parameters
QDA.claculate_mean()
QDA.claculate_sigma()
y_pre = QDA.classify(x_test)          # do classify after training
QDA_acc = (y_pre == y_test).mean()
print ('accuracy of QDA is:{:.2f}'.format(QDA_acc))