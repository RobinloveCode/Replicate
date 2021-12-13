# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:24:25 2021

@author: ruobi
"""
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


from itertools import product,combinations
import random


import statsmodels.api as sm


from sklearn.linear_model import LinearRegression
import pickle
def get_data(name):
    #file_name = 'C:\\Users\\lenovo\\Desktop\\FuzzyTimeSeries\\pyFTS-master\\pyFTS\\'+name+'.csv'
    file_name = 'D:\\Multivarate paper program\\'+name+'.csv'
    #D:\Multivarate paper program\monthly_data
  
    dat = pd.read_csv(file_name)

    dat = dat.fillna(method='ffill')
    return dat,dat.columns
def format_data(dat,order,idx=0):
    n_sample=dat.shape[0]-max_lag
    x=np.zeros((n_sample,len(order)*dat.shape[1]))
    y=np.zeros((n_sample,1))
    for i in range(n_sample):
        x[i,:]=dat[i:i+max_lag,:][order,:].ravel()
        y[i]  =dat[i+max_lag,idx]
    return x,y
def setup_seed(seed):
     
     np.random.seed(seed)
     random.seed(seed)
def rimse_test(hyper):
    order=hyper[0]
    alpha=hyper[1]
    start=hyper[2]
    feature_idx=hyper[3]
    dat_=np.concatenate((norm_data[:,target_idx].reshape(-1,1),norm_data[:,feature_idx]),axis=1)
    x,y=format_data(dat_,order,target_idx)
    # trainx,trainy=x[:-test_l,:],y[:-test_l,:]
    # trainx=np.concatenate((trainx,np.ones((trainx.shape[0],1))),axis=1)
    # mod = sm.OLS(trainy, trainx)
    # res=mod.fit()
    # print(res.summary())
    # model=LinearRegression()#Ridge(alpha=alpha)
    x,y=format_data(dat_,order,target_idx)
    # trainx,trainy=x[:-test_l,:],y[:-test_l,:]
    # model.fit(trainx,trainy.ravel())
    filename = fname+'.sav'
    # pickle.dump(model, open(filename, 'wb'))\
        
    model = pickle.load(open(filename, 'rb'))
 
    pre=model.predict(x[-test_l:,:])
    df_pre=target_scaler.inverse_transform(pre.reshape(-1,1))
    prediction=df_pre+np_data[-1-test_l:-1,target_idx].reshape(-1,1)
    return prediction
# def rimse_test(hyper):
#     order=hyper[0]
#     alpha=hyper[1]
#     start=hyper[2]
#     feature_idx=hyper[3]
#     dat_=np.concatenate((norm_data[start:,target_idx].reshape(-1,1),norm_data[start:,feature_idx]),axis=1)
#     x,y=format_data(dat_,order,target_idx)
#     trainx,trainy=x[:-test_l,:],y[:-test_l,:]
#     trainx=np.concatenate((trainx,np.ones((trainx.shape[0],1))),axis=1)
#     mod = sm.OLS(trainy, trainx)
#     res=mod.fit()
#     # print(res.summary())
#     model=LinearRegression()#Ridge(alpha=alpha)
#     x,y=format_data(dat_,order,target_idx)
#     trainx,trainy=x[:-test_l,:],y[:-test_l,:]
#     model.fit(trainx,trainy.ravel())
#     filename = fname+'.sav'
#     # pickle.dump(model, open(filename, 'wb'))\
        
#     model = pickle.load(open(filename, 'rb'))
#     print(model.coef_,model.intercept_)
#     print(len(model.coef_),x.shape,model.intercept_)
#     pre=model.predict(x[-test_l:,:])
#     df_pre=target_scaler.inverse_transform(pre.reshape(-1,1))
#     prediction=df_pre+np_data[-1-test_l:-1,target_idx].reshape(-1,1)
#     return prediction
def round_(x):
    fn=x-int(x)
    if 0.75>fn >=0.25:
        # print('if')
        x=int(x)+0.5
    elif fn>=0.75:
       
        x=int(x)+1
  
    else:
        x=int(x)
    return x
def rmse(targets, forecasts):
    """
    Root Mean Squared Error

    :param targets: 
    :param forecasts: 
    :return: 
    """
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(forecasts, list):
        forecasts = np.array(forecasts)
    return np.sqrt(np.nanmean((targets - forecasts) ** 2))
def MASE(actual, pred, Scale):
    '''
    MASE = 1/n * sum_{i=1}^{n}|pred_i - actual_i} |/ sum_traning(|diff|)
    input: actual and pred should be np.array
    output: MASE

    '''
    assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
    MAE =  1/len(actual) *np.linalg.norm(actual - pred,1)
    # Scale =  1/(len(history)-1) * np.linalg.norm(np.diff(history),1)
    MASE = MAE/Scale

    return MASE
def RMSE(actual, pred):
    """
    RMSE = sqrt(1/n * sum_{i=1}^{n}{pred_i - actual_i} )
    input: actual and pred should be np.array
    output: RMSE
    """
    assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
    RMSE = np.sqrt( 1/len(actual) *np.linalg.norm(actual - pred,2)**2)
    return RMSE
def MAPE(actual, pred):
    '''
    MAPE = 1/n * sum_{i=1}^{n} |pred_i - actual_i} |/|actual_i|
    input: actual and pred should be np.array
    output: MAPE

    '''
    assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
    MAPE =  1/len(actual) *np.linalg.norm((actual - pred)/actual, 1)

    return MAPE
if __name__ == "__main__":
    setup_seed(0)
    
    VLCC_v=[
     'VLCC 315-320K DWT Newbuilding Prices',
    
        'VLCC D / H 310K DWT 5 Year old Secondhand Prices',
       
        'VLCC Scrap Value',
      'UL / VLCC Orderbook',
     'UL / VLCC Orderbook.1',
       'UL / VLCC Orderbook.2',
        'VLCC Fleet Growth',
        'LIBOR Interest Rates'
    ]
    Aframax_v=[
             'Aframax Tanker 113-115K DWT Newbuilding Prices',
                'Aframax D / H 105K DWT 5 Year Old Secondhand Prices',
                  'Aframax Scrap Value',
              'Aframax Tanker Orderbook',
             'Aframax Tanker Orderbook.1',
                'Aframax Tanker Orderbook.2',
       
'LIBOR Interest Rates'
             ]
    Suezmax_v=[
            
             'Suezmax Tanker 156-158K DWT Newbuilding Prices',
                'Suezmax D / H 160K DWT 5 Year Old Secondhand Prices',
                    'Suezmax Scrap Value',
                   
                'Suezmax Orderbook',
              'Suezmax Orderbook.1',
                'Suezmax Orderbook.2',
                    'Suezmax Fleet Growth',
                   'LIBOR Interest Rates'
             ]
  
    newdf=pd.read_csv('DATA.csv')
  
    target_idx=0
    features=VLCC_v
    feanames=['VLCC','Aframax','Suezmax']
    best_hypers=[((0, 11), 0, 4, (1, 2, 3, 4)),((0, 11), 0, 8, (1, 2, 3)),((8, 9, 11), 0, 12, (6,))
        ]
    jjj=0
    scales=[1.2807881773399015, 0.7413793103448276, 0.8472906403940886]
    for features in [VLCC_v,Aframax_v,Suezmax_v]:
        
        Scale=scales[jjj]
        fname=feanames[jjj]
        rimse_best_hyper=best_hypers[jjj]
        jjj+=1
        
        features_comb=[]
        for i in range(1,len(features)):
            a=list(combinations(list(range(1,len(features))),i))
            for j in a:
                features_comb.append(j)
        orders=[]
        max_lag=12#12#11 T-1 10 T-2 0 T-12
     
        lags=[0,6,7,8,9,10,11]
        # lags=[0,12,23,22,21,20]
        for i in range(1,4):        
            # a=list(combinations(list(range(max_lag)),i))
            a=list(combinations(list(lags),i))
            for j in a:
                if  11 in j :
                    orders.append(j)
  
        np_data=newdf[features].values[:,:]
        dif_data=np_data[1:,:]-np_data[:-1,:]
        validation_l=24#48
        test_l=24#48
        """
        cross validation
        """
        # scaler=preprocessing.MinMaxScaler()
        # scaler.fit(dif_data[:-validation_l-test_l,:])  
        # target_scaler=preprocessing.MinMaxScaler()
        # target_scaler.fit(dif_data[:-validation_l-test_l,target_idx].reshape(-1,1)) 
        # norm_data=scaler.transform(dif_data)
        # n_features=norm_data.shape[1]
        # starts=list(range(0,40,4))#[0,12,24]#,36]
        
       
        # alphas=[0]#np.arange(0,31)/10#[0,0.05,0.1,0.15,0.2,0.25]#[0]#[0.1,0.05,0.02,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.9,1]41/80
        # rimse_hypers=list(product(orders,alphas,starts,features_comb))
        # rimse_cv_loss=list(map(rimse_cv,rimse_hypers[:]))
        # rimse_idx=rimse_cv_loss.index(min(rimse_cv_loss))
        # rimse_best_hyper= rimse_hypers[rimse_idx]
        
        """
        test
        """
     
        # scaler.fit(dif_data[:-test_l,:])  
        filename = fname+'scaler'+'.sav'
        scaler=pickle.load(open(filename, 'rb'))
        norm_data=scaler.transform(dif_data) 
        # pickle.dump(scaler, open(filename, 'wb'))
        # target_scaler.fit(dif_data[:-test_l,target_idx].reshape(-1,1)) 
        filename = fname+'tscaler'+'.sav'
        target_scaler=pickle.load(open(filename, 'rb'))
        # pickle.dump(target_scaler, open(filename, 'wb'))
        
       
        
        rimse_prediction=rimse_test(rimse_best_hyper)
        for i in range(len(rimse_prediction)):
            rimse_prediction[i]=round_(rimse_prediction[i])
        rimse_rmse=RMSE(np_data[-test_l:,target_idx].reshape(-1,1),rimse_prediction)
        rimse_mape=MAPE(np_data[-test_l:,target_idx].reshape(-1,1),rimse_prediction)
      
       
        rimse_mase=MASE(np_data[-test_l:,target_idx].ravel(),rimse_prediction.ravel(),Scale)
       
        print(fname,'rimse',rimse_rmse,'mase',rimse_mase,'mape',rimse_mape)