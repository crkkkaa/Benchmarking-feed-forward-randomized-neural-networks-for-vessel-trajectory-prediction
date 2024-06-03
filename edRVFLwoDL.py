

from itertools import product,combinations
import ForecastLib
from sklearn import preprocessing
import numpy as np
import pandas as pd
import torch,random
import torch.nn as nn
import torch.nn.functional as  F
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
from torch.optim.lr_scheduler import ExponentialLR,MultiStepLR
from sklearn.linear_model import Ridge,Lasso
import os
def sigmoid(x):

    z = 1/(1 + np.exp(-x)) 
    return z
def relu(X):
   return np.maximum(0,X)
class edELM(object):
    def __init__(self, Nu,Nh, input_scale,alpha=0.1,nlayer=1,verbose=0,act='sigmoid'):
        
        self.act=act
        self.Nu = Nu # number of inputs
        self.Nh = Nh # number of units per layer
        self.alpha=alpha
        self.nlayer=nlayer
        # sparse recurrent weights init
        self.Win = [np.random.uniform(-input_scale, input_scale, size=(Nu,Nh))]# for i in range(self.nlayer)]
        for i in range(self.nlayer-1):
            self.Win.append(np.random.uniform(-input_scale, input_scale, size=(Nh,Nh)))
            
        
        self.Bias = np.zeros((Nh,1))
            
       
    
    def computeState(self,x_raw):
        #Win Nh*Nu
        #x_raw Nu*Nsampl
        #inistate= Nh*1
        Ns=x_raw.shape[0]
        states = []#np.zeros((self.Nh*self.nlayer,Ns))
        # print(x_raw.shape)
   
        # self.Win[layer] = self.Win[layer][:,:x_raw.shape[0]+1]
        # state=x_raw
        if self.act=='sigmoid':
            for i in range(self.nlayer):
                
                if i==0:
                    state=sigmoid(x_raw.dot(self.Win[i]))
                    # print(state.shape,x_raw.shape)
                else:
                    x_=state#np.concatenate((x_raw,state),axis=1)
                    state=sigmoid(x_.dot(self.Win[i]))

                states.append(state)
            
        else:
            for i in range(self.nlayer):
                state=relu(state.dot(self.Win[i]))
                states[i*self.Nh:(i+1)*self.Nh,:]=state.T
        # print(state.shape,x_raw.shape)
        return states#n_sample,n_h
    def fit(self,x,y):
        states=self.computeState(x)
        
        self.models=[]
        for i in range(self.nlayer):
            state=states[i]
            # print(state.shape,x.shape)
            cat=state#np.concatenate([state,x],axis=1)
            model=Ridge(alpha=self.alpha)
            model.fit(cat,y)
            self.models.append(model)
    def predict(self,x):
        states=self.computeState(x)
        predictions=[]
        for i in range(self.nlayer):
            state=states[i]
            cat=state#np.concatenate([state,x],axis=1)
        # print(cat.shape,x.shape)
            prediction=self.models[i].predict(cat)
            predictions.append(prediction)
            # print('p',prediction.shape)
        prediction=np.concatenate(predictions,axis=1)
        stacked_predictions = np.stack(predictions, axis=-1)
        final_prediction = np.mean(stacked_predictions, axis=-1)
        # print(prediction.shape,np.mean(prediction,axis=1).shape,x.shape)
        return final_prediction

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


def EDELM_pre(x,h,l,insample=False):    
    global dat
    global target
    #chs,pools,orders
    n_node=h[0]
    alpha=h[1]
    order=h[2]
    iss=h[3]
    nlayer=h[4]
    # Nu,Nh, input_scale,alpha
    allx,ally=format_data(dat,target,order,step)
    trainx,trainy=allx[:-l,:],ally[:-l,:]
    testx,testy=allx[-l:,:],ally[-l:,:]
    m=edELM(allx.shape[1],n_node, iss,alpha,nlayer=nlayer)
    model=m.fit(trainx,trainy)

    if insample is True:
        # print('mlp',allx.shape)
        _pre_=m.predict(allx)
    else:
        _pre_=m.predict(testx)
    return _pre_,model





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
        
def cv(hypers,X,Y):
    #x norm_dif[:-test_l,:]
    #idx target_idx
    #l prediction horizon
    kf = KFold(n_splits=5,shuffle=False)
    losses=[]
    for h in hypers:
        loss=0
        for train_index, test_index in kf.split(X):

            X_train, X_val = X[train_index,:], X[test_index,:]
            y_train, y_val = Y[train_index,:], Y[test_index,:]
            y_hat,_=EDELM_pre(X_val,h, len(test_index))
            print(y_hat)
            for i in range(y_hat.shape[1]):

                e=compute_err(y_val[:,i], y_hat[:,i])['RMSE']
                loss+=e
        losses.append(loss)
    return hypers[losses.index(min(losses))]
def get_data(name):
    #file_name = D:\YangSibo
    file_name = name+'.csv' 
    # file_name=name
    dat = pd.read_csv(file_name)
    dat = dat.fillna(method='ffill')
    return dat,dat.columns
data_path = r"xx"
result_path = r"xx"
step=1
variables=['Latitude','Longitude','Speed','Course']
target_variables=['Latitude','Longitude']
def process_allfile(file_path):
    global dat
    global target
    metric=ForecastLib.TsMetric()
    
    order=6
    orders=[order]
    
    Nhs=[2]
    
    regs=[2**i for i in range(-5,6)]
    iss=[1]
    Nl=[4]
    hypers=list(product(Nhs,regs,orders,iss,Nl))
    
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
    
    np_data=dat
    scaler=preprocessing.MinMaxScaler()
    scaler.fit(np_data[:-vall-testl,:])  
    norm_data=scaler.transform(np_data)
    
    _best_h=cv(hypers,normtrainx, normtrainy)
    seeds = 10  
    all_predictions = []
    for seed in range(seeds):
        np.random.seed(seed)
        norm_pres,_=EDELM_pre(norm_data, _best_h, testl)
        predictions=yscaler.inverse_transform(norm_pres)
        if dif1st:
            predictions=predictions.ravel()+target[-testl-1:-1]
        all_predictions.append(predictions)
    elmpre=np.concatenate(all_predictions,axis=1)
    elmpre=pd.DataFrame(elmpre)
    output_file = os.path.join(result_path, os.path.basename(file_path).replace('.csv', '.csv'))
    elmpre.to_csv(output_file)


for file in os.listdir(data_path):
    if file.endswith('.csv'):
        file_path = os.path.join(data_path, file)
        process_allfile(file_path)