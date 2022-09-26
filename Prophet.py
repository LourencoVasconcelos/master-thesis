# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 02:28:26 2022

@author: loure
"""

from darts import TimeSeries
from darts.models import Prophet
import pandas as pd
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import statsmodels.api as stm
import itertools
from sklearn import preprocessing

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
#standardize option
# =============================================================================
# m = df['Waiting_Time'].mean()
# s = df['Waiting_Time'].std()
# df['Waiting_Time'] = (df['Waiting_Time']-m)/s
# ============================================================================

#%%

# Create a TimeSeries, specifying the time and value columns
# series1 = TimeSeries.from_dataframe(df1, 'Acquisition_Time', 'Waiting_Time',freq='D')
# series2 = TimeSeries.from_dataframe(df2, 'Acquisition_Time', 'Waiting_Time',freq='D')
# series3 = TimeSeries.from_dataframe(df3, 'Acquisition_Time', 'Waiting_Time',freq='D')
# series4 = TimeSeries.from_dataframe(df4, 'Acquisition_Time', 'Waiting_Time',freq='D')


# Set aside the last 4 months as a validation series
# Dataset has 17 months, take last 4
# train1, val1 = series1[:-100], series1[-100:]
# train2, val2 = series2[:-100], series2[-100:]
# train3, val3 = series3[:-100], series3[-100:]
# train4, val4 = series4[:-100], series4[-100:]


holidays_ = pd.DataFrame({
  'holiday': 'feriados',
  'ds': pd.to_datetime(['2017-12-01', '2017-12-08','2017-12-25','2018-01-01',
                        '2018-03-30','2018-04-01', '2018-04-25', '2018-05-01',
                        '2018-05-31','2018-06-10','2018-08-15','2018-10-05',
                        '2018-11-01','2018-12-01','2018-12-08','2018-12-25',
                        '2019-01-01','2019-04-19','2019-04-21','2019-04-25']),})


def run_Prophet(train,val,series):
    model = Prophet(daily_seasonality = True, 
                    weekly_seasonality= True,  holidays = holidays_,
                    growth="linear",
                    seasonality_mode="additive")
    model.fit(train)
    prediction = model.predict(len(val))
    #series.plot()
    #prediction.plot(label='forecast', low_quantile=0.05, high_quantile=0.95)
    #plt.legend()
    
    predictions = prediction._xa.values.flatten()
    val_predictions = val._xa.values.flatten()
    error = sm.mean_absolute_error(val_predictions, predictions)
    print("Prophet Mean Absolute Error: "+str(error)+"  PARAMETERS: "+str(len(model.model_params)))
    return error
    
    
#run_Prophet(train1,val1,series1)
#run_Prophet(train2,val2,series2)
#run_Prophet(train3,val3,series3)
#run_Prophet(train4,val4,series4)

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
        error_sns1 = run_Prophet(train1, val1, series1)
        error_sns2 = run_Prophet(train2, val2, series2)
        error_sns3 = run_Prophet(train3, val3, series3)
        error_sns4 = run_Prophet(train4, val4, series4)
        mae1+= error_sns1
        mae2+= error_sns2
        mae3+= error_sns3
        mae4+= error_sns4
    
    print("PROPHET SNS1 Mean Absolute Error TOTAL: "+str(mae1))
    print("PROPHET SNS2 Mean Absolute Error TOTAL: "+str(mae2))
    print("PROPHET SNS3 Mean Absolute Error TOTAL: "+str(mae3))
    print("PROPHET SNS4 Mean Absolute Error TOTAL: "+str(mae4))
    print("PROPHET SNS1 Mean Absolute Error: "+str(mae1/5))
    print("PROPHET SNS2 Mean Absolute Error: "+str(mae2/5))
    print("PROPHET SNS3 Mean Absolute Error: "+str(mae3/5))
    print("PROPHET SNS4 Mean Absolute Error: "+str(mae4/5))

get_maes(df1,df2,df3,df4)

#RESULTS

# PROPHET SNS1 Mean Absolute Error: 35.39594065842345
# PROPHET SNS2 Mean Absolute Error: 25.645770318799812
# PROPHET SNS3 Mean Absolute Error: 20.642552555092642
# PROPHET SNS4 Mean Absolute Error: 9.767041241278218


#%% 100 days prevision - just test stuff

# Previsao 100 dias:
# Prophet SNS1 Mean Absolute Error: 45.32781127573591 
# Prophet SNS2 Mean Absolute Error: 21.744018315496106
# Prophet SNS3 Mean Absolute Error: 18.751090371589736
# Prophet SNS4 Mean Absolute Error: 8.009872161750929 