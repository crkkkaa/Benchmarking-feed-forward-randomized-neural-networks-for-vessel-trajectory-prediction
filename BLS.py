# -*- coding: utf-8 -*-

from easydict import EasyDict
import ForecastLib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from BLS_Regression import bls_regression
import ForecastLib
from sklearn import preprocessing
import os

def compute_error(actual,prediction):
    actual=actual.ravel()
    prediction=prediction.ravel()
    # history=history.ravel()
    metric=ForecastLib.TsMetric()
    _mase=metric.MAE(actual, prediction)#, history)
    _rmse=metric.RMSE(actual, prediction)
    _mape=metric.MAPE(actual, prediction)
    _err={'MAE':_mase,'MAPE':_mape,'RMSE':_rmse}
    return _err   
def get_swap_dict(d):
    return {v: k for k, v in d.items()}
def get_naive(ts_data,order,step):
    """
    ts_data: n_time*n_features
    assume the first column is the target
    """
    time,n_feature=ts_data.shape[0],ts_data.shape[1]
    n_sample=time-order-step+1
    # x=np.zeros((n_sample,order*n_feature))
    y=np.zeros((n_sample,step))
    for i in range(n_sample):
        # x[i,:]=ts_data[i:i+order,:].ravel()
        n=ts_data[i+order-1,0]
        y[i,:]=n
    return y
def format_data_flatten(ts_data,order,step):
    """
    ts_data: n_time*n_features
    assume the first column is the target
    """
    time,n_feature=ts_data.shape[0],ts_data.shape[1]
    n_sample=time-order-step+1
    x=np.zeros((n_sample,order*n_feature))
    y=np.zeros((n_sample,1))
    for i in range(n_sample):
        x[i,:]=ts_data[i:i+order,:].ravel()
        y[i,:]=target[i+order+step-1]
    return x,y
def format_data(dat,target,order,step):
    n_sample=dat.shape[0]-order-step+1
    x=np.zeros((n_sample,dat.shape[1]*order))
    y=np.zeros((n_sample,2))
    for i in range(n_sample):
        x[i,:]=dat[i:i+order,:].ravel()
        y[i,:]  =target[i+order+step-1,:]
    return x,y
def BLS_pre(BLSX_trv,BLSy_trv ,Xtest,attr):
    # order=hyper[0]
    input_size=BLSX_trv.shape[-1]
    output_size=step
   
    s=attr.s
    C=attr.C
    NumFea=attr.NumFea
    NumWin=attr.NumWin
    NumEnhan=attr.NumEnhan
    results=bls_regression(BLSX_trv,BLSy_trv ,Xtest,norm_evaly,s,C,NumFea,NumWin,NumEnhan)
    ky_hat=results[-1]
    # print(ky_hat.shape,Xtest.shape)
    return ky_hat

data_path = r"xx"
result_path = r"xx"

def process_allfile(file_path):
    data = pd.read_csv(file_path)
    variables=['Latitude','Longitude','Speed','Course']
    target_variables=['Latitude','Longitude']
    
    dif1st=False    
    step=1
    testl=int(len(data)*0.2)
    test_data_df=data[variables][-testl:]
    train_data_df=data[variables][:-testl]
    train_data=train_data_df.values
    test_data=test_data_df.values

    testl=len(test_data)
    trainl=len(train_data)
    vall=int(train_data.shape[0]*0.2)
        
    target=data[target_variables].values.astype(np.float64)
    dat=data[variables].values.astype(np.float64)
        
        
    x_=dat[:-testl,:]
    y_=target[:-testl]

    if dif1st:
        dif_dat=dat[1:-testl,:]-dat[:-testl-1,:]
        dif_target=target[1:-testl].reshape(-1,1)-target[:-testl-1].reshape(-1,1)
        



    xscaler=preprocessing.MinMaxScaler()
    xscaler.fit(dat[:-testl,:])
    normx_=xscaler.transform(dat)
        
    yscaler=preprocessing.MinMaxScaler()
    yscaler.fit(target[:-testl])
    normy_=yscaler.transform(target)

    norm_train_data=normx_[:-testl]
    norm_test_data=normx_[-testl:]
    norm_trainy=normy_[:-testl]
    norm_testy=normy_[-testl:]

    normy_=yscaler.transform(target)
    norm_testy=normy_[-testl:]
    ts_data=dat
            
    max_ts=ts_data[:-testl].max(axis=0)
    min_ts=ts_data[:-testl].min(axis=0)


    den_ts=max_ts-min_ts
    ts_norm=(ts_data-min_ts[None,:])/den_ts[None,:]


    max_target = target[:-testl].max(axis=0)
    min_target = target[:-testl].min(axis=0)
    den_target = max_target - min_target
    target_norm = (target - min_target[None, :]) / den_target[None, :]



    blspre=[]
    for seed in np.arange(10):
        np.random.seed(seed)
        Cs=[2**-5,2**-4,2**-3,2**-2,2**-1,2**0,2**1,2**2,2**3,2**4,2**5,2**6]
           
        s=seed
        NumFeas=[2,4,8]
        NumWins=[2,4,8]
        NumEnhans=[2,4,8]
        orders=[6]
        min_loss=np.inf
           
            
        for order in orders:
            allx,ally=format_data(normx_,normy_,order,step)
            norm_trainx,norm_trainy=allx[:trainl],ally[:trainl]
            
            kf = KFold(n_splits=5)
            order_loss=[]
            for C in Cs:
                for NumFea in NumFeas:
                    for NumWin in NumWins:
                        for NumEnhan in NumEnhans[:1]:
                                
                            loss=0
                            attr=EasyDict()
                            attr['order']=order
                            attr['C']=C
                            attr['s']=seed
                            attr['randseed']=seed
                            attr['NumFea']=NumFea
                            attr['NumEnhan']=NumEnhan
                            attr['NumWin']=NumWin
                               
                            for i, (train_index, val_index) in enumerate(kf.split(norm_trainx)):
                                k_trainx,k_trainy=norm_trainx[train_index,:],norm_trainy[train_index,:]
                                k_valx,k_valy=norm_trainx[val_index,:],norm_trainy[val_index,:]
                                s=attr.s
                                C=attr.C
                                NumFea=attr.NumFea
                                NumWin=attr.NumWin
                                NumEnhan=attr.NumEnhan
                                    
                                results=bls_regression(k_trainx,k_trainy ,k_valx,k_valy,seed,C,NumFea,NumWin,NumEnhan)
                                print(results)
                                ky_hat=results[-1].A
                                    
                                err=compute_error(ky_hat.ravel(),k_valy.ravel())['RMSE']
                                loss+=err
                            if loss<min_loss:
                                min_loss=loss
                                best_attr=attr
            import pickle
            with open('Hyper/BLS'+'.pickle', 'wb') as handle:
                pickle.dump(best_attr, handle, protocol=pickle.HIGHEST_PROTOCOL)


        
        best_attr.randseed=seed
        s=seed
        C=best_attr.C
        NumFea=best_attr.NumFea
        NumWin=best_attr.NumWin
        NumEnhan=best_attr.NumEnhan
                
        results=bls_regression(norm_train_data,norm_trainy,norm_test_data,norm_testy,seed,C,NumFea,NumWin,NumEnhan)
        ytest_hat=results[-1].A
        yhat=(ytest_hat * den_target) + min_target
        blspre.append(yhat)

    blspre=np.concatenate(blspre,axis=1)
    blspre=pd.DataFrame(blspre)
    output_file = os.path.join(result_path, os.path.basename(file_path))
    blspre.to_csv(output_file)
    
    
for file in os.listdir(data_path):
    if file.endswith('.csv'):
        file_path = os.path.join(data_path, file)
        process_allfile(file_path)