
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:58:27 2020

@author: andreadellavecchia
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:36:01 2020

@author: andreadellavecchia
"""

import sklearn.metrics.pairwise
import numpy as np
import time
from liblinear.liblinearutil import train, predict
import pandas as pd
import seaborn as sns
import pickle


def calculate_embedd_blocks(X,y,m,k='rbf'):
    start = time.time()
    
    X_tilde=X[:m]
    y_tilde=y[:m]
    X_compl=X[m:]
    y_compl=y[m:]
    del X
    del y
    
    K_M=sklearn.metrics.pairwise.pairwise_kernels(X_tilde,metric=k)
    #end = time.time()
    #K_M_time=end-start
    #assert(np.all(K_M>=0)==True)
    assert(K_M.shape[0]==X_tilde.shape[0])
    #assert(np.allclose(K_M,np.transpose(K_M))==True)
    #print("time to build K_M =",K_M_time)


    #start = time.time()
    [D,U]=np.linalg.eigh(K_M)
    #end = time.time()
    #diag_time=end-start
    assert(D.shape[0]==K_M.shape[0])
    assert(U.shape==K_M.shape)
    #print("time to diagonlize =",diag_time)
    
    # # rebuild K_M_root_inv
    # start = time.time()
    # K_M_root_inv=np.dot(U,np.dot(np.diag(D**(-0.5)),np.transpose(U)))
    # end = time.time()
    # K_M_root_inv_time=end-start
    # print("time to rebuild K_M^(-1/2)=",K_M_root_inv_time)
    
    
    # check decomposition
    #assert(np.allclose(K_M,np.dot(U,np.dot(np.diag(D),np.transpose(U))))==True)   
    del K_M
    
    # calulate square root of D 
    D_root=D**(0.5)
    del D
    
    X_hat1=np.dot(U,np.dot(np.diag(D_root),np.transpose(U)))
    del y_tilde
    
    #start = time.time()
    K_compl=sklearn.metrics.pairwise.pairwise_kernels(X_compl,X_tilde,metric=k)
    #end = time.time()
    #K_compl_time=end-start
    #assert(np.all(K_compl>=0)==True)
    assert(K_compl.shape==(X_compl.shape[0],X_tilde.shape[0]))
    del X_tilde
    del X_compl
    #print("time to build K_nM =",K_compl_time)
    
    #start = time.time()
    
    X_hat2=np.dot(K_compl,np.dot(U,np.dot(np.diag(D_root**(-1)),np.transpose(U))))
    del y_compl

    end = time.time()
    multiplication_time=end-start
    print("time = ",multiplication_time)
    
    #print("total time=",K_M_time+K_compl_time+diag_time+multiplication_time)
    return np.vstack((X_hat1,X_hat2))


def exec(X,y,m_list,lambda_list):
    output=pd.DataFrame(columns=['m','acc','\u03BB'])
    #output=pd.DataFrame(columns=['m','c_err','\u03BB'])
    i=0
    split=int(X.shape[0]*0.8)
    for lam in lambda_list:
        for m in m_list:
            X_m=calculate_embedd_blocks(X,y,m)
            model = train(y[:split], X_m[:split], '-s 2 -c {}'.format(1./(2.*lam)))
            p_label, p_acc, p_val = predict(y[split:], X_m[split:], model)
            #line to calculate c-err
            #c_err = (1 - p_acc[0]) / 100
            output.loc[i]=[m,p_acc[0],'\u03BB={}'.format(lam)]
            #output.loc[i]=[m,c_err,'\u03BB={}'.format(lam)]
            i+=1
    output.to_csv('./SUSY_s3_mult_C.csv')
    print(output)
    f=sns.lineplot(x=output.m,y=output.acc,hue=output['\u03BB'],
                    style=output['\u03BB'], markers=True, dashes=False)
    #f=sns.lineplot(x=output.m,y=output.c_err,hue=output['\u03BB'],
     #           style=output['\u03BB'], markers=True, dashes=False)
    with open("SUSY_s3_mult_C.pkl", 'wb') as out: 
        pickle.dump(f, out)
    fig = f.get_figure()
    fig.savefig("SUSY_s3_mult_C.png")
    return

def exec2(X,y,m_list,lambda_list):
    output=np.zeros((len(lambda_list),len(m_list)))
    for i,m in enumerate(m_list):
        for j,lam in enumerate(lambda_list):
            print("m =",m)
            X_m=calculate_embedd_blocks(X,y,m)
            model = train(y[:80000], X_m[:80000], '-s 3 -c {}'.format(1./(2.*lam)))
            p_label, p_acc, p_val = predict(y[80000:], X_m[80000:], model)   
            output[j,i]=p_acc[0]
    output=pd.DataFrame(output,index=lambda_list,columns=m_list)
    print(output)
    pl=sns.heatmap(output)
    fig = pl.get_figure()
    fig.savefig("SUSY_heatplot_s3.png")
    return 
            
    
        
        
            
    
    
    
