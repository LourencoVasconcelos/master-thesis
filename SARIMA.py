# -*- coding: utf-8 -*-
"""
@author: Lourenço Vasconcelos
"""


import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
import sklearn.metrics as sm
import itertools
import statsmodels.api as stm
from darts.models import ARIMA
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
series1 = TimeSeries.from_dataframe(df1, 'Acquisition_Time', 'Waiting_Time',freq='D')
series2 = TimeSeries.from_dataframe(df2, 'Acquisition_Time', 'Waiting_Time',freq='D')
series3 = TimeSeries.from_dataframe(df3, 'Acquisition_Time', 'Waiting_Time',freq='D')
series4 = TimeSeries.from_dataframe(df4, 'Acquisition_Time', 'Waiting_Time',freq='D')


# Set aside the last 4 months as a validation series
# Dataset has 17 months, take last 4
train1, val1 = series1[:-1], series1[-1:]
train2, val2 = series2[:-1], series2[-1:]
train3, val3 = series3[:-1], series3[-1:]
train4, val4 = series4[:-1], series4[-1:]


def get_best_model(train, val, df):
    p = range(0,5)
    d= range(0,2)
    q = range(0,5)
    pdq = [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]
    
    ps = range(0,3)
    ds = range(0,2)
    qs = range(0,3)
    s = range(2,5)
    seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(ps, ds, qs, s))] #specify the seasonal order here


    aics = []
    bics = []
    orders = []
    s_orders = []
    
    train_y = train._xa.values.flatten()
    #val_y = val._xa.values.flatten()
    train_y = df["Waiting_Time"].to_numpy()
    df_x = df.drop(columns=["Waiting_Time"])
    le = preprocessing.LabelEncoder()
    df_x['Season'] = df_x['Season'].replace(df_x.Season.values, le.fit_transform(df_x.Season.values))
    df_x=df_x.drop(columns=["Acquisition_Time"])
    #model = stm.tsa.arima.ARIMA(endog=train_y,exog=df1_x, order=(0,0,0))
    k=0
    for i in pdq:
        print(k)
        for j in seasonal_pdq:
            k=k+1
            try:
                model = stm.tsa.statespace.SARIMAX(endog=train_y, exog=df_x, order=i, seasonal_order=j)
                results = model.fit(disp=0)
            except:
                continue
            #print('order:',i, 'AIC: ', results.aic)
            aics.append(results.aic)
            bics.append(results.bic)
            orders.append(i)
            s_orders.append(j)
    
    return orders[aics.index(min(aics))], s_orders[aics.index(min(aics))]

# print("hi")
# min1, min1s = get_best_model(train1,val1,df1)
# print("SNS 1 DONE")
# min2, min2s = get_best_model(train2,val2,df2)
# print("SNS 2 DONE")
# min3, min3s = get_best_model(train3,val3,df3)
# print("SNS 3 DONE")
# min4, min4s = get_best_model(train4,val4,df4)
# print("SNS 4 DONE")
# print("Best Model for SNS 1: " + str(min1) + " seasonal: " + str(min1s)) # (1,1,2)
# print("Best Model for SNS 2: " + str(min2) + " seasonal: " + str(min2s)) # (4,1,7)
# print("Best Model for SNS 3: " + str(min3) + " seasonal: " + str(min3s)) # (1,1,9)
# print("Best Model for SNS 4: " + str(min4) + " seasonal: " + str(min4s)) # (1,1,3)

# =============================================================================
# Best Model for SNS 1: (1, 1, 2) seasonal: (0, 1, 1, 4)
# Best Model for SNS 2: (1, 1, 1) seasonal: (1, 1, 2, 4)
# Best Model for SNS 3: (3, 0, 3) seasonal: (0, 1, 2, 4)
# Best Model for SNS 4: (1, 0, 2) seasonal: (0, 1, 1, 4)
# =============================================================================
def best_model(series,train, val, p,d,q, s_order):
    model = ARIMA(p,d,q, seasonal_order=s_order)
    model.fit(train)
    prediction = model.predict(len(val))
    # series.plot()
    # prediction.plot(label='forecast', low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    predictions = prediction._xa.values.flatten()
    val_predictions = val._xa.values.flatten()
    error = sm.mean_absolute_error(val_predictions, predictions)
    #print("NUMBER OF PARAMETERS:  " + str(len(model.model_params)))
    return error



def get_maes(df1,df2,df3,df4):
    mae1 = 0
    mae2 = 0
    mae3 = 0
    mae4 = 0
    for i in range(8,13):

        series1 = TimeSeries.from_dataframe(df1, 'Acquisition_Time', 'Waiting_Time',freq='D')
        series2 = TimeSeries.from_dataframe(df2, 'Acquisition_Time', 'Waiting_Time',freq='D')
        series3 = TimeSeries.from_dataframe(df3, 'Acquisition_Time', 'Waiting_Time',freq='D')
        series4 = TimeSeries.from_dataframe(df4, 'Acquisition_Time', 'Waiting_Time',freq='D')
        train1, val1 = series1[:-i], series1[-i:]
        train2, val2 = series2[:-i], series2[-i:]
        train3, val3 = series3[:-i], series3[-i:]
        train4, val4 = series4[:-i], series4[-i:]
        error_sns1 = best_model(series1, train1, val1,p=1,d=1,q=2,s_order=(0,1,1,4))
        error_sns2 = best_model(series2, train2, val2,p=1,d=1,q=1, s_order=(1,1,2,4))
        error_sns3 = best_model(series3, train3, val3,p=3,d=0,q=3, s_order=(0,1,2,4))
        error_sns4 = best_model(series4, train4, val4,p=1,d=0,q=2, s_order=(0,1,1,4))
        mae1+= error_sns1
        mae2+= error_sns2
        mae3+= error_sns3
        mae4+= error_sns4
    
    print("SARIMA SNS1 Mean Absolute Error TOTAL: "+str(mae1))
    print("SARIMA SNS2 Mean Absolute Error TOTAL: "+str(mae2))
    print("SARIMA SNS3 Mean Absolute Error TOTAL: "+str(mae3))
    print("SARIMA SNS4 Mean Absolute Error TOTAL: "+str(mae4))
    print("SARIMA SNS1 Mean Absolute Error: "+str(mae1/5))
    print("SARIMA SNS2 Mean Absolute Error: "+str(mae2/5))
    print("SARIMA SNS3 Mean Absolute Error: "+str(mae3/5))
    print("SARIMA SNS4 Mean Absolute Error: "+str(mae4/5))

get_maes(df1,df2,df3,df4)
