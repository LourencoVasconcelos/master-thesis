# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 17:01:20 2022

@author: loure
"""


from darts import TimeSeries
from darts.models import RNNModel
import pandas as pd
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt
from darts.metrics import mae
import sklearn.metrics as sm
import itertools
import numpy as np

# Read a pandas DataFrame
df1 = pd.read_csv('sns dataset/sns1_index.csv', delimiter=",")
df2 = pd.read_csv('sns dataset/sns2_index.csv', delimiter=",")
df3 = pd.read_csv('sns dataset/sns3_index.csv', delimiter=",")
df4 = pd.read_csv('sns dataset/sns4_index.csv', delimiter=",")

df1.rename(columns= {'Unnamed: 0':'Acquisition_Time'}, inplace=True)
df2.rename(columns= {'Unnamed: 0':'Acquisition_Time'}, inplace=True)
df3.rename(columns= {'Unnamed: 0':'Acquisition_Time'}, inplace=True)
df4.rename(columns= {'Unnamed: 0':'Acquisition_Time'}, inplace=True)

df1 = df1.drop(columns=["Emergency_Stage","People_Waiting"])
df2 = df2.drop(columns=["Emergency_Stage","People_Waiting"])
df3 = df3.drop(columns=["Emergency_Stage","People_Waiting"])
df4 = df4.drop(columns=["Emergency_Stage","People_Waiting"])

df1 = df1.groupby(["Acquisition_Time","Season"], as_index=False).mean()
df2 = df2.groupby(["Acquisition_Time","Season"], as_index=False).mean()
df3 = df3.groupby(["Acquisition_Time","Season"], as_index=False).mean()
df4 = df4.groupby(["Acquisition_Time","Season"], as_index=False).mean()


# Create a TimeSeries, specifying the time and value columns
series1 = TimeSeries.from_dataframe(df1, 'Acquisition_Time', 'Waiting_Time',freq='D')
series2 = TimeSeries.from_dataframe(df2, 'Acquisition_Time', 'Waiting_Time',freq='D')
series3 = TimeSeries.from_dataframe(df3, 'Acquisition_Time', 'Waiting_Time',freq='D')
series4 = TimeSeries.from_dataframe(df4, 'Acquisition_Time', 'Waiting_Time',freq='D')


# Set aside the last 4 months as a validation series
# Dataset has 17 months, take last 4
train1, val1 = series1[:-8], series1[-8:]
train2, val2 = series2[:-8], series2[-8:]
train3, val3 = series3[:-8], series3[-8:]
train4, val4 = series4[:-8], series4[-8:]


import torch.nn as nn
    
    

def run_RNN(train, val, series, n_layers=1, n_dropout=0, model_="LSTM", hid_dim=20,
            t_length=7, in_chunk_length=7,rand_state=42,lr=1e-5 , loss = "MSE" ):
    # Normalize the time series 
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)
    #series_transformed = transformer.transform(series)
    
    # create month and year covariate series
    year_series = datetime_attribute_timeseries(
        pd.date_range(start=series.start_time(), freq=series.freq_str, periods=1000),
        attribute="year", one_hot=False,)
    year_series = Scaler().fit_transform(year_series)
    month_series = datetime_attribute_timeseries(
        year_series, attribute="month", one_hot=True)
    covariates = year_series.stack(month_series)
    #cov_train, cov_val = covariates[:-100], covariates[-100:]
    
    
    ll = nn.MSELoss()
    if loss == "l1": 
        ll = nn.L1Loss()
       
    model = RNNModel( model=model_, hidden_dim=hid_dim, dropout=n_dropout, batch_size=8,
        n_epochs=300, optimizer_kwargs={"lr": lr}, model_name="Air_RNN",
        log_tensorboard=True,random_state=rand_state,training_length=t_length, 
        input_chunk_length=in_chunk_length, force_reset=True, save_checkpoints=True, loss_fn= ll,pl_trainer_kwargs={
      "accelerator": "gpu",
      "gpus": [0]
    }
    )   
    model.fit(train_transformed, future_covariates=covariates,
                 val_series=val_transformed, val_future_covariates=covariates,
                 verbose=True)
    
    pred_series = model.predict(n=1, future_covariates=covariates) #, future_covariates=covariates
    #plt.figure(figsize=(8, 5))
    #series_transformed.plot(label="actual")
    #pred_series.plot(label="forecast")
    #plt.title("MAE: {:.2f}%".format(mae(pred_series, val_transformed)))
    #print("RNN MAE: " + str(mae(pred_series, val_transformed)))
    #plt.legend()

    p = transformer.inverse_transform(pred_series)._xa.values.flatten()
    v = transformer.inverse_transform(val_transformed)._xa.values.flatten()
    b = v[0]
    aux = [b]
    #c = p[-1]
    #aux2 = [c]
    
    t = np.array(aux)
    #r = np.array(aux2)
    mae_l = sm.mean_absolute_error(p,t)
    print(v[0])
    print(p)
    #print(model.output_chunk_length)
    
    return mae_l
    
def get_model_mae_cv(df, layers, dim, learning_r, model, loss_):
    mae_v = 0
    for i in range(8,13):
        series = TimeSeries.from_dataframe(df, 'Acquisition_Time', 'Waiting_Time',freq='D')
        train, val = series[:-i], series[-i:]
        error_sns = run_RNN(train,val,series, n_layers = layers, hid_dim = dim, lr = learning_r, model_= model, loss=loss_)
        mae_v+=error_sns
    final_mae = mae_v/5
    if model == "LSTM":
        print("LSTM Mean Absolute Error: "+str(final_mae))
    else:
        print("GRU Mean Absolute Error: "+str(final_mae))
    return final_mae


def get_best_model(df, _model="LSTM"):
    n_layers = range(1, 5) 
    n_dropout= [0,0.15,0.3] 
    hid_dim = [5,20,50,200]
    in_chunk_length = [7, 12,14,16]
    lr = [1e-3,1e-4,1e-5]
    loss = ["l1","MSE"]
    combinations = [(x[0],x[1],x[2], x[3], x[4], x[5]) for x in list(itertools.product(n_layers,hid_dim,lr, loss, n_dropout, in_chunk_length))]
    best_mae = 1000
    best_x = (0,0,0)
    for x in combinations:
        #mae_v= run_RNN(train,val,series,n_layers=x[0],n_dropout=x[1],hid_dim=x[2],lr=x[3],model_=_model, loss=x[4])
       # mae_v= run_RNN(train,val,series,n_layers=x[0],n_dropout=0,hid_dim=x[1],lr=x[2],model_=_model, loss=x[3])
        mae_v = get_model_mae_cv(df, dim=x[1], learning_r=x[2], model=_model, loss_=x[3],
                                 n_dropout= x[4], in_chunk_length=x[5])
        if mae_v < best_mae:
            best_mae = mae_v
            best_x = x
    
    return best_mae, best_x
    
#run_RNN(train1,val1,series1,n_layers=1, n_dropout=0, hid_dim=50,lr=0.001, model_="LSTM", loss="l1")
#run_RNN(train2,val2,series2,n_layers=1, n_dropout=0, hid_dim=200,lr=0.0001, model_="LSTM", loss="l1")
#run_RNN(train3,val3,series3,n_layers=1, n_dropout=0, hid_dim=20,lr=0.001, model_="LSTM", loss="MSE")
#run_RNN(train4,val4,series4,n_layers=1, n_dropout=0, hid_dim=5,lr=1e-05, model_="LSTM", loss="l1")



# print("START")
# best_mae_sns1,best_combination_sns1 = get_best_model(df1)
# print("SNS 1 DONE")
# best_mae_sns2,best_combination_sns2 = get_best_model(df2)
# print("SNS 2 DONE")
# best_mae_sns3,best_combination_sns3 = get_best_model(df3)
# print("SNS 3 DONE")
# best_mae_sns4,best_combination_sns4 = get_best_model(df4)
# print("SNS 4 DONE")
# print("BEST LSTM COMBINATION: " + str(best_combination_sns1) +" BEST MAE SNS1: " + str(best_mae_sns1))
# print("BEST LSTM COMBINATION: " + str(best_combination_sns2) +" BEST MAE SNS2: " + str(best_mae_sns2))
# print("BEST LSTM COMBINATION: " + str(best_combination_sns3) +" BEST MAE SNS3: " + str(best_mae_sns3))
# print("BEST LSTM COMBINATION: " + str(best_combination_sns4) +" BEST MAE SNS4: " + str(best_mae_sns4))
    
# BEST LSTM COMBINATION: (1, 5, 0.001, 'l1') BEST MAE SNS1: 24.76111128376256
# BEST LSTM COMBINATION: (1, 5, 1e-05, 'MSE') BEST MAE SNS2: 11.086476049190598
# BEST LSTM COMBINATION: (1, 5, 0.001, 'MSE') BEST MAE SNS3: 7.134976807522389
# BEST LSTM COMBINATION: (1, 200, 0.001, 'l1') BEST MAE SNS4: 10.761825284409966
    
#%%   GRU
# print("START GRU")
# best_mae_sns1_gru,best_combination_sns1_gru = get_best_model(df1,_model="GRU")
# print("SNS 1 END")
# best_mae_sns2_gru,best_combination_sns2_gru = get_best_model(df2,_model="GRU")
# print("SNS 2 END")
# best_mae_sns3_gru,best_combination_sns3_gru = get_best_model(df3,_model="GRU")
# print("SNS 3 END")
# best_mae_sns4_gru,best_combination_sns4_gru = get_best_model(df4,_model="GRU")

# print("BEST LSTM COMBINATION: " + str(best_combination_sns1) +" BEST MAE SNS1: " + str(best_mae_sns1))
# print("BEST LSTM COMBINATION: " + str(best_combination_sns2) +" BEST MAE SNS2: " + str(best_mae_sns2))
# print("BEST LSTM COMBINATION: " + str(best_combination_sns3) +" BEST MAE SNS3: " + str(best_mae_sns3))
# print("BEST LSTM COMBINATION: " + str(best_combination_sns4) +" BEST MAE SNS4: " + str(best_mae_sns4))
# print("Best MAE SNS1: " + str(best_mae_sns1_gru)+ " best GRU combination: " + str(best_combination_sns1_gru))
# print("Best MAE SNS2: " + str(best_mae_sns2_gru)+ " best GRU combination: " + str(best_combination_sns2_gru))
# print("Best MAE SNS3: " + str(best_mae_sns3_gru)+ " best GRU combination: " + str(best_combination_sns3_gru))
# print("Best MAE SNS4: " + str(best_mae_sns4_gru)+ " best GRU combination: " + str(best_combination_sns4_gru))


#GRU MODELS
# Best MAE SNS1: 26.799188268839565 best GRU combination: (1, 50, 0.0001, 'l1')
# Best MAE SNS2: 12.393537177841315 best GRU combination: (1, 50, 1e-05, 'MSE')
# Best MAE SNS3: 7.6012640088592125 best GRU combination: (1, 5, 0.001, 'MSE')
# Best MAE SNS4: 14.414814828781775 best GRU combination: (1, 200, 0.0001, 'l1')

#%% RUN MODELS AND GET MAE VALUES

def get_single_mae(df, layers, h_dim, learning_r, model, loss_, dropout=0):
    mae_v = 0
    for i in range(8,13):
        series = TimeSeries.from_dataframe(df, 'Acquisition_Time', 'Waiting_Time',freq='D')
        train, val = series[:-i], series[-i:]
        if model == "LSTM":
            error_sns = run_RNN(train,val,series,n_layers=layers, n_dropout=dropout,
                                hid_dim=h_dim,lr=learning_r, model_="LSTM", loss= loss_)
        if model == "GRU":
            error_sns = run_RNN(train,val,series, n_layers= layers, n_dropout=dropout, 
                                hid_dim= h_dim, lr= learning_r, model_="GRU", loss =loss_)
        mae_v+=error_sns
        
    return mae_v/5
        

def get_maes(df1,df2,df3,df4,m):
    mae1 = 0
    mae2 = 0
    mae3 = 0
    mae4 = 0

    for i in range (8,13): 
        series1 = TimeSeries.from_dataframe(df1, 'Acquisition_Time', 'Waiting_Time',freq='D')
        series2 = TimeSeries.from_dataframe(df2, 'Acquisition_Time', 'Waiting_Time',freq='D')
        series3 = TimeSeries.from_dataframe(df3, 'Acquisition_Time', 'Waiting_Time',freq='D')
        series4 = TimeSeries.from_dataframe(df4, 'Acquisition_Time', 'Waiting_Time',freq='D')
        train1, val1 = series1[:-i], series1[-i:]
        train2, val2 = series2[:-i], series2[-i:]
        train3, val3 = series3[:-i], series3[-i:]
        train4, val4 = series4[:-i], series4[-i:]
        if m == "LSTM":
            error_sns1 = run_RNN(train1,val1,series1,n_layers=1, n_dropout=0, hid_dim=5,lr=0.001, model_="LSTM", loss="l1")
            error_sns2 = run_RNN(train2,val2,series2,n_layers=1, n_dropout=0, hid_dim=5,lr=1e-05, model_="LSTM", loss="MSE")
            error_sns3 = run_RNN(train3,val3,series3,n_layers=1, n_dropout=0, hid_dim=5,lr=0.001, model_="LSTM", loss="MSE")
            error_sns4 = run_RNN(train4,val4,series4,n_layers=1, n_dropout=0, hid_dim=200,lr=0.001, model_="LSTM", loss="l1")
        if m == "GRU":
            error_sns1 = run_RNN(train1,val1,series1, n_layers=1, n_dropout=0, hid_dim=50, lr=0.0001, model_="GRU", loss ="l1")
            error_sns2 = run_RNN(train2,val2,series2, n_layers=1, n_dropout=0, hid_dim=50, lr=0.00001, model_="GRU", loss = "MSE")
            error_sns3 = run_RNN(train3,val3,series3, n_layers=1, n_dropout=0, hid_dim=5, lr=0.0001, model_="GRU", loss = "MSE")
            error_sns4 = run_RNN(train4,val4,series4, n_layers=1, n_dropout=0, hid_dim=200, lr=0.0001, model_="GRU", loss = "l1")
        mae1+= error_sns1
        mae2+= error_sns2
        mae3+= error_sns3
        mae4+= error_sns4
    
    if m == "LSTM":
        print("LSTM SNS1 Mean Absolute Error TOTAL: "+str(mae1))
        print("LSTM SNS2 Mean Absolute Error TOTAL: "+str(mae2))
        print("LSTM SNS3 Mean Absolute Error TOTAL: "+str(mae3))
        print("LSTM SNS4 Mean Absolute Error TOTAL: "+str(mae4))
        print("LSTM SNS1 Mean Absolute Error: "+str(mae1/5))
        print("LSTM SNS2 Mean Absolute Error: "+str(mae2/5))
        print("LSTM SNS3 Mean Absolute Error: "+str(mae3/5))
        print("LSTM SNS4 Mean Absolute Error: "+str(mae4/5))
    if m =="GRU":
        print("GRU SNS1 Mean Absolute Error TOTAL: "+str(mae1))
        print("GRU SNS2 Mean Absolute Error TOTAL: "+str(mae2))
        print("GRU SNS3 Mean Absolute Error TOTAL: "+str(mae3))
        print("GRU SNS4 Mean Absolute Error TOTAL: "+str(mae4))
        print("GRU SNS1 Mean Absolute Error: "+str(mae1/5))
        print("GRU SNS2 Mean Absolute Error: "+str(mae2/5))
        print("GRU SNS3 Mean Absolute Error: "+str(mae3/5))
        print("GRU SNS4 Mean Absolute Error: "+str(mae4/5))

#get_maes(df1,df2,df3,df4,"LSTM")
get_maes(df1,df2,df3,df4,"GRU")


#BEST LSTM AND GRU COMBINATIONS
# BEST LSTM COMBINATION: (1, 5, 0.001, 'l1') BEST MAE SNS1: 24.76111128376256
# BEST LSTM COMBINATION: (1, 5, 1e-05, 'MSE') BEST MAE SNS2: 11.086476049190598
# BEST LSTM COMBINATION: (1, 5, 0.001, 'MSE') BEST MAE SNS3: 7.134976807522389
# BEST LSTM COMBINATION: (1, 200, 0.001, 'l1') BEST MAE SNS4: 10.761825284409966
# Best MAE SNS1: 26.799188268839565 best GRU combination: (1, 50, 0.0001, 'l1')
# Best MAE SNS2: 12.393537177841315 best GRU combination: (1, 50, 1e-05, 'MSE')
# Best MAE SNS3: 7.6012640088592125 best GRU combination: (1, 5, 0.001, 'MSE')
# Best MAE SNS4: 14.414814828781775 best GRU combination: (1, 200, 0.0001, 'l1')


#%% GARBAGE OLD

# %LSTM MAE SNS1: 38.56219476977281 best combination: (recurrent layers: 1, dropout: 0, hidden dimensions: 5, learning rate: 1e-05, number of time steps fed: 12) L1Loss 
# %LSTM MAE SNS2: 19.067951640000423 best combination: (recurrent layers: 1, dropout: 0, hidden dimensions: 50, learning rate: 1e-05, number of time steps fed: 12, MSELoss )
# %LSTM MAE SNS3: 17.58867141395552 best combination: (recurrent layers: 1, dropout: 0, hidden dimensions: 5, learning rate: 1e-05, number of time steps fed: 12, MSELoss)
# %LSTM MAE SNS4: 8.152826660172076 best combination: (recurrent layers: 1, dropout: 0, hidden dimensions: 5, learning rate: 1e-4, number of time steps fed: 12, L1Loss) 

# %GRU MAE SNS1: 38.2967882461728 best combination: (1, 0, 5, 0.0001, L1Loss)
# %GRU MAE SNS2: 19.32711086186102 best combination: (1, 0, 20, 1e-05, L1Loss)
# %GRU MAE SNS3: 18.026441356017713 best combination: (1, 0, 20, 1e-05, L1Loss)
# %GRU MAE SNS4: 8.097093694335376 best combination: (1, 0, 5, 0.0001, L1Loss)