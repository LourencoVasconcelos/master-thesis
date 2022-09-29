# -*- coding: utf-8 -*-
"""
@author: Lourenço Vasconcelos
"""


import get_results as gr
import main as bpm
import sklearn.metrics as sm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import prepare_data_DLLearner as dl
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, Ridge, RidgeCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import models_after_dllearner as mdl
import models_before_dllearner as mbd
import rules_fromdllearner as rdl
import myRuleGenerator as mrg






    
def get_rules_sns(features_names, sns, simple=False, y=False):
    
    extra = ''
    yesterday =''
    if simple:
        extra='_simple'
    if y:
        yesterday= '_yesterday'
    rules_dl = []
    for i in range(1,21):
        rules_dl = rules_dl + gr.getrules("C:/Users/loure/Desktop/Tese/dataset{s}{o}/sns{k}_ready{n}.csv".format(n=i,k=sns,s=extra,o=yesterday),
                                        "C:/Users/loure/Desktop/Tese/dllearner_datasets{s}{o}/results{n}_sns{k}/target".format(n=i,k=sns,s=extra, o=yesterday))
    extracted = rdl.extractRules_DLLearner(rules_dl) # gets the rules chosen by DL-Learner
    rules = rdl.create_Rules(extracted,features_names)    
    rules = mrg.cleanRules(rules)
    return rules

#%% RUN
# THESIS MODELs

        
def thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names, model,sns_,_simple=False, _y=False):

    #run_dl_learner_sns(df,sns=sns_) 
    rules = get_rules_sns(features_names,sns=sns_,simple=_simple,y=_y)
    
    #standardization
    # scaler = StandardScaler()
    # f = scaler.fit(train_X)
    # train_X = f.transform(train_X)
    # test_X = f.transform(test_X)
    
    if model=="en":
        mdl.elasticNet_afterDL(rules, train_X, train_y, test_X, test_y)
    elif model=="l1":
        mdl.lasso_afterDL(rules, train_X, train_y, test_X, test_y)
    elif model=="l2":
        mdl.ridge_afterDL(rules, train_X, train_y, test_X, test_y)


def thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names, model,_sns,_simple=False,_y=False):

    #run_dl_learner_sns(df,sns="1") 
    rules = get_rules_sns(features_names, sns=_sns, simple=_simple, y=_y)
    #standardization
    # scaler = StandardScaler()
    # f = scaler.fit(train_X)
    # train_X = f.transform(train_X)
    # test_X = f.transform(test_X)
    
    if model=="en":
        mdl.model_afterDL(rules, train_X, train_y, test_X, test_y,l1r=0.5,sns_= _sns)
    elif model=="l1":
        mdl.model_afterDL(rules, train_X, train_y, test_X, test_y,l1r=1,sns_= _sns)
    elif model=="l2":
        mdl.model_afterDL(rules, train_X, train_y, test_X, test_y,l1r=0,sns_= _sns)
 
#%% GET DATA NORMAL WITH COLUMN SERVICE SNS1

# le = preprocessing.LabelEncoder()
# df = pd.read_csv("sns dataset\sns1.csv", index_col=0)
# df= df.drop(columns=['Urgency_Type'])
# df['Service'] = df['Service'].replace(df.Service.values, le.fit_transform(df.Service.values))
# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
# train, test = train_test_split(df, test_size=0.2)
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# # mean = train_y.mean()
# # stdev = train_y.std()
# # train_y = (train_y-mean)/stdev
# test_y = test.Waiting_Time.values
# #test_y = (test_y-mean)/stdev
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns

# # #Run with Service column
# # mbd.run_dl_learner_sns1(df) 
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"en","1")    
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l1","1") 
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l2","1")
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"en","1")    
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l1","1") 
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l2","1")


#%%  Run without Service column, just like TimeSeries datasets SNS1

# le = preprocessing.LabelEncoder()
# df = pd.read_csv("sns dataset\sns1_simple.csv", index_col=0)
# df=df.reset_index()
# df = df.drop(columns=["Acquisition_Time"])
# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
# train, test = train_test_split(df, test_size=0.3)
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# # mean = train_y.mean()
# # stdev = train_y.std()
# # train_y = (train_y-mean)/stdev
# test_y = test.Waiting_Time.values
# #test_y = (test_y-mean)/stdev
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns
# # df['Waiting_Time'] = df['Waiting_Time'].astype(int)
# # a=df['Waiting_Time'].unique()
# # a.sort()
# # print(a)

# #mbd.run_dl_learner_sns1_simple(df)

# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"en","1",_simple=True)    
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l1","1",_simple=True) 
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l2","1",_simple=True)
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"en","1",_simple=True)    
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l1","1",_simple=True) 
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l2","1",_simple=True)





#%% SNS2 Without Service Columns

# le = preprocessing.LabelEncoder()
# df = pd.read_csv("sns dataset\sns2_simple.csv", index_col=0)
# df=df.reset_index()
# df = df.drop(columns=["Acquisition_Time"])
# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
# df = df.drop(df.index[94])
# train, test = train_test_split(df, test_size=0.3)
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# test_y = test.Waiting_Time.values
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns
# #Remove row 94 por ser outlier demasiado fora do resto e nao funciona ao criar labels para o dl_learner
# # df['Waiting_Time'] = df['Waiting_Time'].astype(int)
# # a=df['Waiting_Time'].unique()
# # a.sort()
# # print(a)

# #mbd.run_dl_learner_sns_simple(df,"2")

# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"en","2",_simple=True)    
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l1","2",_simple=True) 
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l2","2",_simple=True)
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"en","2",_simple=True)    
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l1","2",_simple=True) 
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l2","2",_simple=True)
    
#%% SNS2 With Service Column   
    
# =============================================================================
# le = preprocessing.LabelEncoder()
# o_df = pd.read_csv("sns dataset\sns2.csv", index_col=0)
# df= o_df.drop(columns=['Urgency_Type'])
# df['Service'] = df['Service'].replace(df.Service.values, le.fit_transform(df.Service.values))
# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
# train, test = train_test_split(df, test_size=0.3)
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# test_y = test.Waiting_Time.values
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns    
    
# # # df['Waiting_Time'] = df['Waiting_Time'].astype(int)
# # # a=df['Waiting_Time'].unique()
# # # a.sort()
# # # print(a)
# # 
# #mbd.run_dl_learner_sns(df,"2")

# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"en","2",_simple=True)    
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l1","2",_simple=True) 
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l2","2",_simple=True)
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"en","2",_simple=True)    
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l1","2",_simple=True) 
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l2","2",_simple=True)    
# =============================================================================

#%% SNS3 Without Service Columns

# le = preprocessing.LabelEncoder()
# df = pd.read_csv("sns dataset\sns3_simple.csv", index_col=0)
# df=df.reset_index()
# df = df.drop(columns=["Acquisition_Time"])
# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))

# train, test = train_test_split(df, test_size=0.3)
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# test_y = test.Waiting_Time.values
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns

# # df['Waiting_Time'] = df['Waiting_Time'].astype(int)
# # a=df['Waiting_Time'].unique()
# # a.sort()
# # print(a)

# #mbd.run_dl_learner_sns_simple(df,"3")

# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"en","3",_simple=True)    
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l1","3",_simple=True) 
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l2","3",_simple=True)
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"en","3",_simple=True)    
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l1","3",_simple=True) 
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l2","3",_simple=True)

#%% SNS3 With Service Column


# le = preprocessing.LabelEncoder()
# o_df = pd.read_csv("sns dataset\sns3.csv", index_col=0)
# df= o_df.drop(columns=['Urgency_Type'])
# df['Service'] = df['Service'].replace(df.Service.values, le.fit_transform(df.Service.values))
# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
# train, test = train_test_split(df, test_size=0.3)
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# test_y = test.Waiting_Time.values
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns    
    
# # df['Waiting_Time'] = df['Waiting_Time'].astype(int)
# # a=df['Waiting_Time'].unique()
# # a.sort()
# # print(a)

# #mbd.run_dl_learner_sns(df,"3")

# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"en","3",_simple=False)    
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l1","3",_simple=False) 
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l2","3",_simple=False)
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"en","3",_simple=False)    
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l1","3",_simple=False) 
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l2","3",_simple=False)  
# =============================================================================

#%% SNS4 Without Service Column

# le = preprocessing.LabelEncoder()
# df = pd.read_csv("sns dataset\sns4_simple.csv", index_col=0)
# df=df.reset_index()
# df = df.drop(columns=["Acquisition_Time"])
# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))

# train, test = train_test_split(df, test_size=0.3)
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# test_y = test.Waiting_Time.values
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns

# # df['Waiting_Time'] = df['Waiting_Time'].astype(int)
# # a=df['Waiting_Time'].unique()
# # a.sort()
# # print(a)

# # mbd.run_dl_learner_sns_simple(df,"4")

# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"en","4",_simple=True)    
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l1","4",_simple=True) 
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l2","4",_simple=True)
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"en","4",_simple=True)    
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l1","4",_simple=True) 
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"l2","4",_simple=True)


#%% SNS4 With Service Column

# le = preprocessing.LabelEncoder()
# o_df = pd.read_csv("sns dataset\sns4.csv", index_col=0)
# df= o_df.drop(columns=['Urgency_Type'])
# df['Service'] = df['Service'].replace(df.Service.values, le.fit_transform(df.Service.values))
# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
# train, test = train_test_split(df, test_size=0.3)
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# test_y = test.Waiting_Time.values
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns    
    
# df['Waiting_Time'] = df['Waiting_Time'].astype(int)
# a=df['Waiting_Time'].unique()
# a.sort()
# print(a)

# mbd.run_dl_learner_sns(df,"4")

# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"en","4",_simple=False)    
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l1","4",_simple=False) 
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l2","4",_simple=False)
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"en","4",_simple=False)    
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names, "l1","4",_simple=False) 
# thesis_model_alphas(df,train_X, train_y, test_X,  test_y,features_names,"l2","4",_simple=False)  

#%% SNS 1 With Yesterday

# le = preprocessing.LabelEncoder()
# df = pd.read_csv("sns dataset\sns1_simple2.csv", index_col=0)

# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
# train, test = train_test_split(df, test_size=0.3)
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# test_y = test.Waiting_Time.values
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns


# mbd.run_dl_learner_sns_simple_yesterday(df,"1")

# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"en","1",_simple=True, _y=True)    
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l1","1",_simple=True, _y=True) 
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l2","1",_simple=True, _y=True)
#thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"en","1",_simple=True, _y=True)    
# thesis_model_alphas(df,train_X, train_y, test_X, test_y,features_names,"l1","1",_simple=True, _y=True) 
# thesis_model_alphas(df,train_X, train_y, test_X, test_y,features_names,"l2","1",_simple=True, _y=True)


#################################################################################################################
#SNS 2

# le = preprocessing.LabelEncoder()
# df = pd.read_csv("sns dataset\sns2_simple2.csv", index_col=0)

# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
# train, test = train_test_split(df, test_size=0.3)
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# test_y = test.Waiting_Time.values
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns


#mbd.run_dl_learner_sns_simple_yesterday(df,"2")
############# CHECK THIS
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"en","2",_simple=True, _y=True)    
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l1","2",_simple=True, _y=True) 
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l2","2",_simple=True, _y=True)
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"en","2",_simple=True, _y=True)    
# thesis_model_alphas(df,train_X, train_y, test_X, test_y,features_names,"l1","2",_simple=True, _y=True) 
# thesis_model_alphas(df,train_X, train_y, test_X, test_y,features_names,"l2","2",_simple=True, _y=True)


#################################################################################################################
#SNS 3

# le = preprocessing.LabelEncoder()
# df = pd.read_csv("sns dataset\sns3_simple2.csv", index_col=0)

# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
# train, test = train_test_split(df, test_size=0.3)
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# test_y = test.Waiting_Time.values
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns


# mbd.run_dl_learner_sns_simple_yesterday(df,"3")

# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"en","3",_simple=True, _y=True)    
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l1","3",_simple=True, _y=True) 
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l2","3",_simple=True, _y=True)
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"en","3",_simple=True, _y=True)    
# thesis_model_alphas(df,train_X, train_y, test_X, test_y,features_names,"l1","3",_simple=True, _y=True) 
# thesis_model_alphas(df,train_X, train_y, test_X, test_y,features_names,"l2","3",_simple=True, _y=True)


#################################################################################################################
# SNS 4

# le = preprocessing.LabelEncoder()
# df = pd.read_csv("sns dataset\sns4_simple2.csv", index_col=0)

# df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
# train, test = train_test_split(df, test_size=0.3)
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# test_y = test.Waiting_Time.values
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns


# mbd.run_dl_learner_sns_simple_yesterday(df,"4")

# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"en","4",_simple=True, _y=True)    
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l1","4",_simple=True, _y=True) 
# thesis_model_cv(df,train_X, train_y, test_X, test_y, features_names,"l2","4",_simple=True, _y=True)
# thesis_model_alphas(df,train_X, train_y, test_X, test_y, features_names,"en","4",_simple=True, _y=True)    
# thesis_model_alphas(df,train_X, train_y, test_X, test_y,features_names,"l1","4",_simple=True, _y=True) 
# thesis_model_alphas(df,train_X, train_y, test_X, test_y,features_names,"l2","4",_simple=True, _y=True)