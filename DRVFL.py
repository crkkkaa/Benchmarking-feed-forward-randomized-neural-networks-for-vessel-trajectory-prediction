

from DeepRVFL_.DeepRVFL import DeepRVFL
# import DeepRVFL
from utils import MSE, config_MG, load_MG
from sklearn import preprocessing
from DeepRVFL_.DeepRVFL import DeepRVFL
from DeepRVFL_.DeepRVFL2 import DeepRVFL2

import pandas as pd
import numpy as np
from itertools import product,combinations
from sklearn.linear_model import Ridge
import ForecastLib
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
import os
from DeepRVFL_.DeepRVFL import DeepRVFL
# import DeepRVFL
from utils import MSE, config_MG, load_MG
from sklearn import preprocessing
from DeepRVFL_.DeepRVFL import DeepRVFL
from DeepRVFL_.DeepRVFL2 import DeepRVFL2

import pandas as pd
import numpy as np
from itertools import product,combinations
from sklearn.linear_model import Ridge
import ForecastLib
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
# from skfeature.function.similarity_based import reliefF,lap_score
def compute_error(actuals,predictions,history=None):
    actuals=actuals.ravel()
    predictions=predictions.ravel()
    
    metric=ForecastLib.TsMetric()
    error={}
    error['RMSE']=metric.RMSE(actuals, predictions)
    # error['MAPE']=metric.MAPE(actuals,predictions)
    error['MAE']=metric.MAE(actuals,predictions)
    if history is not None:
        history=history.ravel()
        error['MASE']=metric.MASE(actuals,predictions,history)    
    return error
def format_data(dat,target,order,step):
    n_sample=dat.shape[0]-order-step+1
    x=np.zeros((n_sample,dat.shape[1]*order))
    y=np.zeros((n_sample,2))
    for i in range(n_sample):
        x[i,:]=dat[i:i+order,:].ravel()
        y[i,:]  =target[i+order+step-1,:]
    return x.T,y.T
def format_disc_data(dat,target,order,max_lag=12,step=1):
    n_sample=dat.shape[0]-max_lag-step+1
    x=np.zeros((n_sample,order*dat.shape[1]))
    y=np.zeros((n_sample,1))
    for i in range(n_sample):
        x[i,:]=dat[i:i+max_lag,:][order,:].ravel()
        y[i]  =target[i+max_lag+step-1,0]
    return x.T,y.T

def round_(x):
    l=len(x)
    r=[]
    for i in range(l):
        ele=x[i]
        int_=int(ele)
        if int_-0.25<ele<=int_+0.25:
            r.append(int_)
        elif int_+0.25<ele<=int_+0.75:
            r.append(int_+0.5)
        else:
            r.append(int_+1)
        # print(ele,r[i])
    return np.array(r)
class Struct(object): pass
def config_load(iss,IP_indexes):

    configs = Struct()
    
    
    configs.iss = iss 
    
    configs.IPconf = Struct()
    configs.IPconf.DeepIP = 0 
    


    configs.enhConf = Struct()
    configs.enhConf.connectivity = 1 
    
    configs.readout = Struct()
    configs.readout.trainMethod = 'Ridge' 
    
    return configs 
def select_indexes(data, indexes):


    return data[:,indexes]
def dRVFL_predict(hyper,data,train_idx,test_idx,layer,s,last_states=None):
    # print(s)
    np.random.seed(s)
    Nu=datastr.inputs.shape[0]

    Nh = hyper[0][0] 

    Nl = layer # 
    
    reg=[]

    iss=[]
    for h in hyper:
        reg.append( h[1])        
        iss.append(h[2])
    configs=config_load(iss,train_idx)
    deepRVFL = DeepRVFL(Nu, Nh, Nl, configs)
    train_targets = select_indexes(datastr.targets, train_idx)
    
    if Nl==1:
        
        states = deepRVFL.computeLayerState(0,datastr.inputs)
    else:
        
        states=deepRVFL.computeLayerState(Nl-1,datastr.inputs,last_states)

    train_states = select_indexes(np.concatenate([states,datastr.inputs],axis=0), train_idx)
    test_states = select_indexes(np.concatenate([states,datastr.inputs],axis=0), test_idx)

    deepRVFL.trainReadout(train_states[:,:], train_targets, reg[-1])
 
    test_outputs_norm = deepRVFL.computeOutput(test_states[:,:]).T

    return test_outputs_norm,states[:,:]


def cross_validation(data,raw_data,train_idx,val_idx,Nl,scaler=None,s=0,boat=50):
    best_hypers=[]
    np.random.seed(s)
    layer_s=None
    for i in range(Nl):
   
        layer=i+1
        layer_h,layer_s=layer_cross_validation(data,raw_data,train_idx,val_idx,layer,
                            scaler=scaler,s=s,last_states=layer_s,best_hypers=best_hypers.copy(),boat=boat)
        
        Nhs=[layer_h[0]]
       
        best_hypers.append(layer_h)

    
   
    return best_hypers 
def layer_cross_validation(data,raw_data,train_idx,val_idx,layer,
                           scaler=None,s=0,last_states=None,best_hypers=None,boat=50):
    cvloss=[]
    np.random.seed(s)
    states=[]
    if layer==1:
        Nhs=[1]
        regs=regs=[2**i for i in range(-5,6)]
        input_scales=[1]
    else:

        input_scales=[1]

        Nhs=[1]
        regs=regs=[2**i for i in range(-5,6)]

    best_loss=np.inf
    for Nh in Nhs[:]:
        for reg in regs[:]:
            for iss in input_scales:
                ar={'layer': layer,'data': data,'raw_data': raw_data,'last_states': last_states,
                    'scaler':scaler,'s':s,'val_idx':val_idx,'train_idx':train_idx,
                    'best_hypers':best_hypers,'Nhs':Nh,'regs':reg,'input_scale':iss}
                closs=layer_obj(ar)
                if closs<best_loss:
                    best_loss=closs
                    args=ar
    
    best_hyper=[args['Nhs'],args['regs'],args['input_scale']]#,args['ratios']]
   
    if layer>1:

            hyper_=best_hypers.copy()#
            hyper_.append(best_hyper)

    else:

        hyper_=[best_hyper]
    _,best_state=dRVFL_predict(hyper_,data,train_idx,val_idx,layer,
                                         s,last_states=last_states)

    return best_hyper,best_state 
def layer_obj(args):
    layer=args['layer']
    best_hypers=args['best_hypers']
    # print('layer',best_hypers)
    #Nhs,regs,input_scale,ratios
    hyper=[args['Nhs'],args['regs'],args['input_scale']]
    data=args['data']
    train_idx,val_idx=args['train_idx'],args['val_idx']
    scaler=args['scaler']
    s=args['s']
    raw_data,last_states=args['raw_data'],args['last_states']
    if layer>1:
#          
            hyper_=[i for i in best_hypers]
            hyper_.append(hyper)

    else:
        hyper_=[hyper]
 
    test_outputs_norm,_=dRVFL_predict(hyper_,data,train_idx,val_idx,layer,
                                     s)
    test_outputs=scaler.inverse_transform(test_outputs_norm)
    actuals=raw_data[-len(val_idx):]
    
    test_err=compute_error(actuals,test_outputs,None)
    
    return test_err['MAE']

data_path = r"xx"
result_path = r"xx"

def process_allfile(file_path):
    global datastr 
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
    yscaler.fit(target[:-testl])
    normy_=yscaler.transform(target)
        
        
    order=6
    datastr=Struct()
    datastr.inputs,datastr.targets=format_data(normx_,normy_,order,step=1)
        
    train_l=datastr.inputs.shape[1]-vall-testl

    Nl=4
    train_idx=range(train_l)
    val_idx=range(train_l,train_l+vall)
    test_idx=range(train_l+vall,datastr.inputs.shape[1])
    best_hypers=cross_validation(datastr,target,train_idx,val_idx,Nl,scaler=yscaler,s=0,boat=50)
    print('Test')
    xscaler.fit(dat[:-testl])
    normx_=xscaler.transform(dat)

    yscaler.fit(target[:-testl])
    normy_=yscaler.transform(target)

    seeds = 10  
    all_predictions = []
    for seed in range(seeds):
        np.random.seed(seed)

        test_outputs_norm_mea=dRVFL_predict(best_hypers,datastr,train_idx,test_idx,Nl,seed)

        predictions=yscaler.inverse_transform(test_outputs_norm_mea[0])
        if dif1st:
            predictions=predictions.ravel()+target[-testl-1:-1]
        all_predictions.append(predictions)
    drvflpre=np.concatenate(all_predictions,axis=1)
    drvflpre=pd.DataFrame(drvflpre)
    
    
    output_file = os.path.join(result_path, os.path.basename(file_path))
    drvflpre.to_csv(output_file)


for file in os.listdir(data_path):
    if file.endswith('.csv'):
        file_path = os.path.join(data_path, file)
        process_allfile(file_path)