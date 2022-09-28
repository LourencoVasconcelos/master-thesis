# -*- coding: utf-8 -*-
"""
@author: Lourenço Vasconcelos
"""

from darts import TimeSeries
from darts.models import TransformerModel
import pandas as pd
import sklearn.metrics as sm
import itertools
import numpy as np
from sklearn import preprocessing
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries

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

#%%

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



def run_Transformer(train,val,series, n_heads=4, n_enc=3, n_dec=3, ffs=512):
    
    
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)
    
    # create month and year covariate series
    year_series = datetime_attribute_timeseries(
        pd.date_range(start=series.start_time(), freq=series.freq_str, periods=529),
        attribute="year", one_hot=False,)
    year_series = Scaler().fit_transform(year_series)
    month_series = datetime_attribute_timeseries(
        year_series, attribute="month", one_hot=True)
    covariates = year_series.stack(month_series)
    cov_train, cov_val = covariates[:-8], covariates[-8:]
    
    model = TransformerModel(input_chunk_length = 7, output_chunk_length= 1,
                             nhead = n_heads, num_encoder_layers=n_enc, num_decoder_layers=n_dec, dim_feedforward=ffs,
                             pl_trainer_kwargs={
                           "accelerator": "gpu", "gpus": [0]}
                            )
    
    model.fit(train_transformed, val_series = val_transformed, verbose = True)
    pred_series = model.predict(n=1)
    
    p = transformer.inverse_transform(pred_series)._xa.values.flatten()
    v = transformer.inverse_transform(val_transformed)._xa.values.flatten()
    b = v[0]
    aux = [b]
    t = np.array(aux)
    
    error= sm.mean_absolute_error(p,t)
    print("Transformer Mean Absolute Error: "+str(error)+"  PARAMETERS: "+str(len(model.model_created)))
    return error
    



def get_model_mae_cv(df, heads, enc, dec, ff):
    mae_v = 0
    for i in range(8,13):
        series = TimeSeries.from_dataframe(df, 'Acquisition_Time', 'Waiting_Time',freq='D')
        train, val = series[:-i], series[-i:]
        error_sns = run_Transformer(train,val,series,n_heads = heads, n_enc=enc, n_dec=dec, ffs= ff)
        mae_v+=error_sns
        
    return mae_v/5


#Function to try different models for 1 step of cross validation, run 5 times
def get_best_model(df):

    heads = [2,4,8]
    enc_dec = range(3,6)
    ff = [256,512,1024,2048]
    

    combinations = [(x[0],x[1],x[1],x[2]) for x in list(itertools.product(heads,enc_dec, ff))]
    best_mae = 1000
    best_x = (0,0,0,0)
    for x in combinations:
        
        mae_v= get_model_mae_cv(df,heads=x[0],enc=x[1],dec=x[1], ff=x[2])
        if mae_v < best_mae:
            best_mae = mae_v
            best_x = x
    
    return best_mae, best_x
    
#run_Transformer(train1,val1,series1)
#run_Transformer(train2,val2,series2)
#run_Transformer(train3,val3,series3)
#run_Transformer(train4,val4,series4)

# print("START")
# best_mae_sns1,best_combination_sns1 = get_best_model(df1)
# print("SNS 1 DONE")
# best_mae_sns2,best_combination_sns2 = get_best_model(df2)
# print("SNS 2 DONE")
# best_mae_sns3,best_combination_sns3 = get_best_model(df3)
# print("SNS 3 DONE")
# best_mae_sns4,best_combination_sns4 = get_best_model(df4)
# print("SNS 4 DONE")
# print("BEST COMBINATION: " + str(best_combination_sns1) +" BEST MAE SNS1: " + str(best_mae_sns1))
# print("BEST COMBINATION: " + str(best_combination_sns2) +" BEST MAE SNS2: " + str(best_mae_sns2))
# print("BEST COMBINATION: " + str(best_combination_sns3) +" BEST MAE SNS3: " + str(best_mae_sns3))
# print("BEST COMBINATION: " + str(best_combination_sns4) +" BEST MAE SNS4: " + str(best_mae_sns4))

###BEST MODEL BY CV and respective MAE
# BEST COMBINATION: (8, 4, 4, 2048) BEST MAE SNS1: 25.437327843749706
# BEST COMBINATION: (2, 5, 5, 1024) BEST MAE SNS2: 14.257237529965545
# BEST COMBINATION: (4, 5, 5, 1024) BEST MAE SNS3: 10.365513226355448
# BEST COMBINATION: (8, 5, 5, 256) BEST MAE SNS4: 13.946626329050375

def get_maes(df1,df2,df3,df4):
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
        error_sns1 = run_Transformer(train1, val1, series1, n_heads=8, n_enc=4, n_dec=4, ffs=2048)
        error_sns2 = run_Transformer(train2, val2, series2, n_heads=2, n_enc=5, n_dec=5, ffs=1024)
        error_sns3 = run_Transformer(train3, val3, series3, n_heads=4, n_enc=5, n_dec=5, ffs=1024)
        error_sns4 = run_Transformer(train4, val4, series4, n_heads=8, n_enc=5, n_dec=5, ffs=256)
        mae1+= error_sns1
        mae2+= error_sns2
        mae3+= error_sns3
        mae4+= error_sns4
    
    print("Transformer SNS1 Mean Absolute Error TOTAL: "+str(mae1))
    print("Transformer SNS2 Mean Absolute Error TOTAL: "+str(mae2))
    print("Transformer SNS3 Mean Absolute Error TOTAL: "+str(mae3))
    print("Transformer SNS4 Mean Absolute Error TOTAL: "+str(mae4))
    print("Transformer SNS1 Mean Absolute Error: "+str(mae1/5))
    print("Transformer SNS2 Mean Absolute Error: "+str(mae2/5))
    print("Transformer SNS3 Mean Absolute Error: "+str(mae3/5))
    print("Transformer SNS4 Mean Absolute Error: "+str(mae4/5))

get_maes(df1,df2,df3,df4)
