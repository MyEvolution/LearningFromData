# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:15:03 2018

@author: XieTianwen
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

num = np.load('num.npy')
'''
your codes
# gy = 
#second_sv = 
'''
gy = np.mat([range(15)]).T
fx = np.mat([range(10)]).T
numx =np.mat( [np.sum(num,axis=1)]).T
numy = np.mat([np.sum(num,axis=0)]).T
numsum = np.sum(num)

#print(numx,numy)
second_sv = []
# get conditional expectation vector, or expectation of g(y)  or f(x)
def Expectation (fxgy,sign,conditional=False):

# a sign indicate what we compute
    if sign == 'x':
        temp=  np.dot(num,fxgy)
        div = numx
    else:
        temp =  np.dot(fxgy.T, num).T
        div = numy
    if conditional == False:
        return temp/div
    else: return np.sum(temp)/numsum
# get expectation of f(x)g(y)
def AllExpectation(fx,gy):
    return np.sum( np.multiply(np.dot(fx,gy.T) , num))/ numsum  
# get B matrix in fact it's B's transpose which the lecture 
def getB():
    pxpy= np.power((numx / numsum) * (numy /numsum).T,0.5)
    pxy = num/numsum
    return pxy/ pxpy
def getEPxy(gy,fx):
    pxy = num/numsum
    return np.sum(np.multiply( fx * gy.T,  pxy))
# from eigenvector and matrix m to get the eigenvalue
def getEigenValue(m,eigenVector):
    return (m*eigenVector)[0][0] / (eigenVector[0][0])

B = getB()

#normalize
fx = fx- Expectation(fx,'y',True)
fx = fx / np.power(Expectation(np.power(fx,2),'y',True),0.5)
gy = gy- Expectation(gy,'x',True)
gy = gy / np.power(Expectation(np.power(gy,2),'x',True),0.5)
last = 0
now = AllExpectation(fx,gy)
i = 0
##ACE algorithm, iteration
while  now - last>0:
    i+=1
    fx = Expectation(gy,'x')
    gy = Expectation(fx,'y')
    #normalize
    fx = fx- Expectation(fx,'y',True)
    fx = fx / np.power(Expectation(np.power(fx,2),'y',True),0.5)
    gy = gy- Expectation(gy,'x',True)
    gy = gy / np.power(Expectation(np.power(gy,2),'x',True),0.5)
    last,now=now,AllExpectation(fx,gy) 
# the singular value is the sqrt of eigenvalue of BB^T 

second_sv = np.sqrt(getEigenValue(B.T*B,np.multiply(gy,np.power(numy/numsum,0.5))))
print('second_sv : {}'.format(second_sv))    
u,s,v = np.linalg.svd(B)
print(s)
print(B*B.T,np.sum(B*B.T,))
print(getEPxy(gy,fx))
plt.plot(np.arange(15), gy, c = 'r')
plt.xlabel('y')
plt.ylabel('gy')
# show the figure
plt.show()