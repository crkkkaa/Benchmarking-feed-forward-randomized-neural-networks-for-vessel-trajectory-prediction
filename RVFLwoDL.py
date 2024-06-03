

import matplotlib.pyplot as plt
from itertools import product,combinations
import sys 
import os
import ForecastLib
from sklearn import preprocessing
import numpy as np
import pandas as pd
import torch,random
import torch.nn as nn
import torch.nn.functional as  F
from sklearn.svm import SVR
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
from torch.optim.lr_scheduler import ExponentialLR,MultiStepLR
from sklearn.linear_model import Ridge,Lasso
from sklearn import linear_model
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
def sigmoid(x):

    z = 1/(1 + np.exp(-x)) 
    return z
def relu(X):
   return np.maximum(0,X)
class ELM(object):
    def __init__(self, Nu,Nh, input_scale,alpha=0.1,verbose=0,act='sigmoid'):
        
        self.act=act
        self.Nu = Nu # number of inputs
        self.Nh = Nh # number of units per layer
        self.alpha=alpha
        # sparse recurrent weights init
        self.Win = np.random.uniform(-input_scale, input_scale, size=(Nu,Nh))
        
        self.Bias = np.zeros((Nh,1))
            
       
    
    def computeState(self,x_raw):
        #Win Nh*Nu
        #x_raw Nu*Nsampl
        #inistate= Nh*1
        Ns=x_raw.shape[0]
        state = np.zeros((self.Nh,Ns))
        # print(x_raw.shape)
   
        # self.Win[layer] = self.Win[layer][:,:x_raw.shape[0]+1]
        if self.act=='sigmoid':
            state=sigmoid(x_raw.dot(self.Win))
            # state=sigmoid(self.Win[:,:].dot(x_raw.T))#+np.tile(np.expand_dims(self.Win[:,-1],1),[1,Ns])
        else:
            state=relu(self.Win[:,:-1].dot(x_raw))+np.tile(np.expand_dims(self.Win[:,-1],1),[1,Ns])
        # print(state.shape,x_raw.shape)
        return state#n_sample,n_h
    def fit(self,x,y):
        state=self.computeState(x)
        # cat=np.concatenate([state,x],axis=1)
        # print(cat.shape)MultiTaskElasticNet

        self.model=Ridge(alpha=self.alpha)
        self.model.fit(state,y)
    def predict(self,x):
        state=self.computeState(x)
        # cat=np.concatenate([state,x],axis=1)
        # print(cat.shape,x.shape)
        prediction=self.model.predict(state)
        return prediction


def compute_err(actual,prediction,history):
    actual=actual.ravel()
    prediction=prediction.ravel()
    history=history.ravel()
    metric=ForecastLib.TsMetric()
    _mase=metric.MAE(actual, prediction)#, history)
    _rmse=metric.RMSE(actual, prediction)
    _mape=metric.MAPE(actual, prediction)
    _err={'MAE':_mase,'MAPE':_mape,'RMSE':_rmse}
    return _err
def generate_batches(x,y,batch_size,shuffle=False):
    """
    x (n_sample,n_feature)
    y (n_sample,n_output)
    """
    n_sample=x.shape[0]
    n_batch=int(n_sample/batch_size)
    if shuffle:
        np.random.shuffle(x)
        np.random.shuffle(y)
    batchx,batchy=[],[]
    for i in range(n_batch):
        bx=x[i*batch_size:(i+1)*batch_size,:]
        by=y[i*batch_size:(i+1)*batch_size,:]
        batchx.append(torch.from_numpy(bx).float())
        batchy.append(torch.from_numpy(by).float())
    if n_batch*batch_size<n_sample:
        batchx.append(torch.from_numpy(x[n_batch*batch_size:,:]).float())
        batchy.append(torch.from_numpy(y[n_batch*batch_size:,:]).float())
    return batchx,batchy

 
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def format_data(dat,target,order,step):
    n_sample=dat.shape[0]-order-step+1
    x=np.zeros((n_sample,dat.shape[1]*order))
    y=np.zeros((n_sample,2))
    for i in range(n_sample):
        x[i,:]=dat[i:i+order,:].ravel()
        y[i,:]  =target[i+order+step-1,:]
    return x,y
def format_data2(dat,dat_target,order,step):
    n_sample=dat.shape[0]-order-step+1
    x=np.zeros((n_sample,dat.shape[1]*order))
    y=np.zeros((n_sample,dat_target.shape[1]))
    for i in range(n_sample):
        x[i,:]=dat[i:i+order,:].ravel()
        y[i]  =dat_target[i+order+step-1,:]
    return x,y
        
def cv(hypers,X,Y):
    kf = KFold(n_splits=5,shuffle=False)
    losses=[]
    for h in hypers:
        loss=0
        for train_index, test_index in kf.split(X):

            X_train, X_val = X[train_index,:], X[test_index,:]
            y_train, y_val = Y[train_index,:], Y[test_index,:]
            y_hat=ELM_pre(X_train,y_train,X_val,h)
            
            for i in range(y_hat.shape[1]):
                e=compute_err(y_val[:,i], y_hat[:,i])['RMSE']
                loss+=e
        losses.append(loss)
    return hypers[losses.index(min(losses))]
def get_data(name):

    file_name = name+'.csv' 

    dat = pd.read_csv(file_name)
    dat = dat.fillna(method='ffill')
    return dat,dat.columns
def ELM_pre(trainx,trainy,testx,hyper):
    nsample=trainx.shape[0]
    order=hyper[0]
    alpha=hyper[1]
    Nh=hyper[2]
    regressor=ELM(trainx.shape[1],Nh,1,alpha)

    regressor.fit(trainx,trainy)
    
    pres=regressor.predict(testx)
    # torch.cuda.empty_cache()
    return pres
def compute_err(actual,prediction):
    actual=actual.ravel()
    prediction=prediction.ravel()
    # history=history.ravel()
    metric=ForecastLib.TsMetric()
    _mase=metric.MAE(actual, prediction)#, history)
    _rmse=metric.RMSE(actual, prediction)
    _mape=metric.MAPE(actual, prediction)
    _err={'MAE':_mase,'MAPE':_mape,'RMSE':_rmse}
    return _err   
    
data_path = r"xx"
result_path = r"xx"

def process_allfile(file_path):
    order=6
    regs=[2**i for i in range(-5,6)]
    Nhs=[3]
    orders=[order]
    hypers=list(product(orders,regs,Nhs))
    
    data=pd.read_csv(file_path)
    variables=['Latitude','Longitude','Speed','Course']
    target_variables=['Latitude','Longitude']
    
    dif1st=False
    
    step=1
    
    testl=int(len(data)*0.2)
    test_data_df=data[variables][-testl:]
    train_data_df=data[variables][:-testl]
    train_data=train_data_df.values
    test_data=test_data_df.values
    
    
    
    min_,max_=train_data.min(axis=0),train_data.max(axis=0)
    den=max_[None,:]-min_[None,:]
    norm_train_data=(train_data-min_)/den
    norm_test_data=(test_data-min_)/den
    
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
    #yscaler.fit(target[:-testl].reshape(-1,1))
    yscaler.fit(target[:-testl])
    normy_=yscaler.transform(target)
    
    testx=dat[-testl:]
    testy=target[-testl:]
    
    
    x,y=format_data(normx_, normy_, order,step)
    normtrainx,normtrainy=x[:-testl,:],y[:-testl,:]
    normtestx=x[-testl:,:]
    
    
    best_hyper= cv(hypers,normtrainx, normtrainy)
    seeds = 10  
    all_predictions = []
    for seed in range(seeds):
        np.random.seed(seed)
        norm_pres=ELM_pre(normtrainx, normtrainy, normtestx, best_hyper)
        predictions=yscaler.inverse_transform(norm_pres)
        if dif1st:
            predictions=predictions.ravel()+target[-testl-1:-1]
        all_predictions.append(predictions)
    elmpre=np.concatenate(all_predictions,axis=1)
    elmpre=pd.DataFrame(elmpre)
    output_file = os.path.join(result_path, os.path.basename(file_path))
    elmpre.to_csv(output_file)
    
for file in os.listdir(data_path):
    if file.endswith('.csv'):
        file_path = os.path.join(data_path, file)
        process_allfile(file_path)