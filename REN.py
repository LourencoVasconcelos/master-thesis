# -*- coding: utf-8 -*-
"""
@author: Lourenço Vasconcelos
"""
import myRuleGenerator as rf
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn.metrics as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning



def GetSignificant_Rules(ls,rules):
    coefs = ls.coef_
    sig_rules = set()
    rule_list = list(rules)
    
    for i_rule in np.arange(len(rule_list)) :
        if coefs[i_rule]!=0 :
            sig_rules.add(rule_list[i_rule])#only important rules
        
    return sig_rules

@ignore_warnings(category=ConvergenceWarning)
def elasticNet_Function(rules, X, y, quantile, alphaL, l1_r =0.5):
    
    X_concat,length, stddev, mean, scale_multipliers = rf.prepareData(rules, X, trim_quantile = quantile)
    regr = ElasticNet(alpha = alphaL,l1_ratio = l1_r, random_state=0,  tol=0.0005, max_iter=5000) #
    regr.fit(X_concat, y)
    #regr = ElasticNetCV()
    
    return regr, stddev, mean, scale_multipliers





def Rules_Accuracy_Rulefit_ElasticNet_ratios(train_X, train_y, features_names, test_X, test_y):

    old_rules = rf.genRules(train_X, train_y, features_names)
    rules = rf.cleanRules(old_rules)
        
    tr = []
    vl = []
    rl = []
    
    def NumberOfRules_Accuracy_graph(l1r):
        train_MAE = []
        val_MAE = []
        rules_length_all = []
        for i in (1,1.1,1.2,1.3,1.4,1.5):
            en, stddev, mean, scale_multipliers = elasticNet_Function(rules, train_X, quantile=0.025, y=train_y, alphaL = 10**i, l1_r = l1r)
            train_predictions, rules_length,_,_,_ = rf.GetPredictions(en, train_X, rules)
            test_predictions, rules_length,_,_,_ = rf.GetPredictions(en, test_X, rules)
            train_MAE.append(round(sm.mean_absolute_error(train_y, train_predictions), 2))
            val_MAE.append(round(sm.mean_absolute_error(test_y, test_predictions), 2))
            rules_length_all.append(len(GetSignificant_Rules(en, rules)))
            
        tr.append(train_MAE)
        vl.append(val_MAE)
        rl.append(rules_length_all)
        
    for i in (0.5,0.6,0.7,0.8,0.9):
        NumberOfRules_Accuracy_graph(l1r=i)  


    
    plt.title("Accuracy by number of rules")
    plt.ylabel("Number of relevant rules")
    plt.xlabel("Mean Absolute Error")
#    plt.plot(tr[0],rl[0], 'g--')
    plt.plot(vl[0],rl[0], 'g')
    #plt.plot(tr[1],rl[1], 'b--')
    plt.plot(vl[1],rl[1], 'b')
    #plt.plot(tr[2],rl[2], 'r--')
    plt.plot(vl[2],rl[2], 'r')
    #plt.plot(tr[3],rl[3], 'y--')
    plt.plot(vl[3],rl[3], 'y')
    #plt.plot(tr[4],rl[4], 'c--')
    plt.plot(vl[4],rl[4], 'c')
    plt.legend(["Validation 0.5", "Validation 0.6", "Validation 0.7", "Validation 0.8", 
                "Validation 0.9"], loc="best")
    # plt.legend(["Train 0.5", "Validation 0.5","Train 0.6", "Validation 0.6", "Train 0.7", "Validation 0.7",
    #             "Train 0.8", "Validation 0.8", "Train 0.9", "Validation 0.9"], loc="upper right")
    plt.savefig("Accuracy_by_Number_Rules_ElasticNet_RatiosX.pdf", format="pdf",bbox_inches="tight")
    plt.show()
    
#Rules_Accuracy_Rulefit_ElasticNet_ratios()





def elasticNet_CV(train_X, train_y, test_X, test_y,features_names):
    #i=0.8
    #regr, stddev, mean, scale_multipliers = en.elasticNet_Function(rules, train_X, quantile=0.025, y=train_y, l1_r = 0.5)
    
    old_rules = rf.genRules(train_X, train_y, features_names)
    rules = rf.cleanRules(old_rules)
    
    X_concat,length, stddev, mean, scale_multipliers = rf.prepareData(rules, train_X) #data ready and standardized
    
    regr = ElasticNetCV(random_state=0,  tol=0.0005, max_iter=10000) #alpha = alphaL,
    regr.fit(X_concat, train_y)
    
    train_predictions, rules_length = rf.GetPredictions(regr, train_X, rules)
    test_predictions, rules_length = rf.GetPredictions(regr, test_X, rules)
    print(round(sm.mean_absolute_error(train_y, train_predictions), 2))
    print(round(sm.mean_absolute_error(test_y, test_predictions), 2))
    print(rules_length)



train_MAE = []
val_MAE = []
rules_length_all = []

def Rules_Accuracy_Rulefit_ElasticNet(train_X, train_y, test_X, test_y, features_names):
    old_rules = rf.genRules(train_X, train_y, features_names)
    rules = rf.cleanRules(old_rules)
    def NumberOfRules_Accuracy_graph():
        #for i in (0.5,1,1.1,1.4,1.45): - SNS with service column
        #for i in (0.3,0.4,0.5,0.6,0.7,0.8,0.82): # SNS 1 and 2 without service column
        #for i in (0.5,1,1.1,1.2,1.25): #SNS2 with service column
        #for i in (0.1,0.2,0.3,0.4,0.5,0.6): #SNS3 without service column
        #for i in (0,0.3,0.6,0.8,1,1.1,1.15): #SNS3 with service column
        #for i in (-0.1,0,0.1,0.2,0.3,0.4,0.45): #SNS4 without service column
        #for i in (0,0.3,0.6,0.7,0.8,0.85,0.95): #SNS4 with service column
        #for i in (0.6,0.7,0.8,0.85,0.9,1,1.1): #SNS 1 and 2 with yesterday
        #for i in (0.3,0.4,0.5,0.6,0.7,0.8,0.9): #SNS 3 with yesterday
        for i in (-0.1,0,0.1,0.2,0.3,0.4,0.45): #SNS 3 with yesterday
            en, stddev, mean, scale_multipliers = elasticNet_Function(rules, train_X, quantile=0.025, y=train_y, alphaL = 10**i, l1_r=0.5)
            train_predictions, rules_length = rf.GetPredictions(en, train_X, rules)
            test_predictions, rules_length = rf.GetPredictions(en, test_X, rules)
            train_MAE.append(round(sm.mean_absolute_error(train_y, train_predictions), 2))
            val_MAE.append(round(sm.mean_absolute_error(test_y, test_predictions), 2))
            
            rules_length_all.append(len(GetSignificant_Rules(en, rules)))
            
            rules_ = rf.get_rules(en.coef_, rules, features_names, stddev, mean, scale_multipliers)
            rules_ = rules_[-rules_length:]
            rules_ = rules_[rules_.coef != 0].sort_values(by="support",ascending=False)
            #print(rules_)
            
            # f = open("rules/elasticnet/elasticnet_rules_{}.txt".format(i), "a")
            # f.write(rules_.to_string())
            # f.close()
            
            
    NumberOfRules_Accuracy_graph()  
    print(train_MAE)
    print(val_MAE)
    print(rules_length_all)
    # plt.title("Accuracy by number of rules")
    # plt.ylabel("Number of relevant rules")
    # plt.xlabel("Mean Absolute Error")
    # plt.plot(train_MAE,rules_length_all,'r--')
    # plt.plot(val_MAE, rules_length_all, 'r')
    # plt.legend(["Train", "Validation"])
    # plt.savefig("Accuracy_ElasticNet.pdf", format="pdf",bbox_inches="tight")
    # plt.show()
        
#Rules_Accuracy_Rulefit_ElasticNet()

train_MAE_i = []
val_MAE_i = []
rules_length_all_i = []

def Rules_Accuracy_Rulefit_ElasticNet_iterative(train_X, train_y, test_X, test_y, features_names):
    old_rules = rf.genRules(train_X, train_y, features_names)
    rules = rf.cleanRules(old_rules)
    def NumberOfRules_Accuracy_graph():
        #for i in (0.5,1,1.1,1.4,1.45): - SNS1 with service column
        #for i in (0.1,0.2,0.3,0.4,0.5,0.6): SNS 1 and 2 without service column
        #for i in (0.5,1,1.1,1.2,1.25):#SNS2 with service column
        #for i in (0,0.1,0.2,0.3,0.4,0.5): #SNS3 without service column
        #for i in (0,0.3,0.6,0.8,1,1.1,1.15): #SNS3 with service column
        #for i in (-0.1,-0.05,0,0.05,0.1,0.15,0.2): #SNS4 without service column
        #for i in (0,0.3,0.6,0.7,0.8,0.85,0.95): #SNS4 with service column
        #for i in (0.2,0.4,0.6,0.8,0.9,1): #SNS 1 and 2 with yesterday
        #for i in (0.2,0.3,0.4,0.5,0.6,0.7): #SNS 3 with yesterday
        for i in (-0.1,-0.05,0,0.05,0.1,0.15,0.2): #SNS 4 with yesterday
            en, stddev, mean, scale_multipliers = elasticNet_Function(rules, train_X, quantile=0.025, y=train_y, alphaL = 10**i)
            new_rules = GetSignificant_Rules(en, rules)
            en2, stddev, mean, scale_multipliers = elasticNet_Function(new_rules, train_X, quantile=0.025, y=train_y, alphaL = 10**i)
            train_predictions, rules_length = rf.GetPredictions(en2, train_X, new_rules)
            test_predictions, rules_length = rf.GetPredictions(en2, test_X,  new_rules)
            train_MAE_i.append(round(sm.mean_absolute_error(train_y, train_predictions), 2))
            val_MAE_i.append(round(sm.mean_absolute_error(test_y, test_predictions), 2))
            rules_length_all_i.append(len(GetSignificant_Rules(en2, new_rules)))
            
            rules_ = rf.get_rules(en2.coef_, new_rules, features_names, stddev, mean, scale_multipliers)
            rules_ = rules_[-rules_length:]
            rules_ = rules_[rules_.coef != 0].sort_values(by="support",ascending=False)
            #print(rules_)
            
            # f = open("rules/elasticnet1/iterative1_rules_{}.txt".format(i), "a")
            # f.write(rules_.to_string())
            # f.close()
            
            
    NumberOfRules_Accuracy_graph()  
    print(train_MAE_i)
    print(val_MAE_i)
    print(rules_length_all_i)
    

train_MAE_2i = []
val_MAE_2i = []
rules_length_all_2i = []
def Rules_Accuracy_Rulefit_ElasticNet_iterative_double(train_X, train_y, test_X, test_y, features_names):
    old_rules = rf.genRules(train_X, train_y, features_names)
    rules = rf.cleanRules(old_rules)
    def NumberOfRules_Accuracy_graph():
        #for i in (0.5,1,1.1,1.4,1.45): - SNS1 with service column
        #for i in (0.1,0.2,0.3,0.4,0.5,0.6): SNS 1 and 2 without service column
        #for i in (0.5,0.9,1.1,1.2,1.28):#SNS2 with service column
        #for i in (0,0.1,0.2,0.3,0.4,0.5): #SNS3 without service column
        #for i in (0,0.3,0.6,0.8,1,1.1,1.15): #SNS3 with service column
        #for i in (-0.3,-0.25,-0.2,-0.15,-0.1,0,0.1): #SNS4 without service column
        #for i in (0,0.3,0.6,0.7,0.8,0.9,1): #SNS4 with service column
        #for i in (0.2,0.4,0.6,0.8,0.9,1): #SNS 1 and 2 with yesterday
        #for i in (0.2,0.3,0.4,0.5,0.6,0.7): #SNS 3 with yesterday
        for i in (-0.3,-0.25,-0.2,-0.15,-0.1,0,0.2): #SNS 4 with yesterday
            en,_,_,_ = elasticNet_Function(rules, train_X, quantile=0.025, y=train_y, alphaL = 10**i)
            new_rules = GetSignificant_Rules(en, rules)
            en2,_,_,_ = elasticNet_Function(new_rules, train_X, quantile=0.025, y=train_y, alphaL = 10**i)
            new_rules = GetSignificant_Rules(en2, new_rules)
            en3, stddev, mean, scale_multipliers = elasticNet_Function(new_rules, train_X, quantile=0.025, y=train_y, alphaL = 10**i)
            train_predictions, rules_length = rf.GetPredictions(en3, train_X, new_rules)
            test_predictions, rules_length= rf.GetPredictions(en3, test_X,  new_rules)
            train_MAE_2i.append(round(sm.mean_absolute_error(train_y, train_predictions), 2))
            val_MAE_2i.append(round(sm.mean_absolute_error(test_y, test_predictions), 2))
            rules_length_all_2i.append(len(GetSignificant_Rules(en3, new_rules)))
            

            
            rules_ = rf.get_rules(en3.coef_, new_rules, features_names, stddev, mean, scale_multipliers)
            rules_ = rules_[-rules_length:]
            rules_ = rules_[rules_.coef != 0].sort_values(by="support",ascending=False)
            #print(rules_)
            
            # f = open("rules/elasticnet2/iterative2_rules_{}.txt".format(i), "a")
            # f.write(rules_.to_string())
            # f.close()
            
    NumberOfRules_Accuracy_graph() 
    print(train_MAE_2i)
    print(val_MAE_2i)
    print(rules_length_all_2i)
    
    
def iterative_elasticnet(t_X, t_y, ts_X, ts_y, fs_ns,sns):
    Rules_Accuracy_Rulefit_ElasticNet(train_X=t_X, train_y=t_y, test_X=ts_X, test_y=ts_y, features_names=fs_ns)
    Rules_Accuracy_Rulefit_ElasticNet_iterative(train_X=t_X, train_y=t_y, test_X=ts_X, test_y=ts_y, features_names=fs_ns)
    Rules_Accuracy_Rulefit_ElasticNet_iterative_double(train_X=t_X, train_y=t_y, test_X=ts_X, test_y=ts_y, features_names=fs_ns)
    
    
    plt.title("Accuracy by number of rules")
    plt.ylabel("Number of relevant rules")
    plt.xlabel("Mean Absolute Error")
    plt.plot(val_MAE,rules_length_all, 'r')
    plt.plot(train_MAE,rules_length_all, 'r--')
    plt.plot(val_MAE_i,rules_length_all_i, 'b')
    plt.plot(train_MAE_i,rules_length_all_i, 'b--')
    plt.plot(val_MAE_2i,rules_length_all_2i, 'g')
    plt.plot(train_MAE_2i,rules_length_all_2i, 'g--')
    
    plt.legend(["Validation ElasticNet", "Train ElasticNet","Validation One repetition", "Train One repetition",
                "Validaton Two repetitions", "Train Two repetitions"], loc="upper right")
    plt.savefig("plots_results_sns{k}/Accuracy_by_Number_Rules_ElasticNet_Iteratives.pdf".format(k=sns), format="pdf",bbox_inches="tight")
    plt.show()
    
def getData(stringDF,sns):
    
    
    if stringDF == "sns_simple2":
        le = preprocessing.LabelEncoder()
        df = pd.read_csv("sns dataset\sns{k}_simple2.csv".format(k=sns), index_col=0)
        df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
        train, test = train_test_split(df, test_size=0.3)
        train_y = train.Waiting_Time.values
        train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
        train_X = train_X.values
        test_y = test.Waiting_Time.values
        test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
        test_X = test_X.values
        train = train.drop(columns=["Waiting_Time","People_Waiting"])
        features_names = train.columns
        
    if stringDF == "sns_simple":
        le = preprocessing.LabelEncoder()
        df = pd.read_csv("sns dataset\sns{k}_simple.csv".format(k=sns), index_col=0)
        df=df.set_index('Emergency_Stage')
        df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
        train, test = train_test_split(df, test_size=0.3)
        train_y = train.Waiting_Time.values
        train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
        train_X=train_X.values
        test_y = test.Waiting_Time.values
        test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
        test_X = test_X.values
        train = train.drop(columns=["Waiting_Time","People_Waiting"])
        features_names = train.columns
    if stringDF == "sns":
        le = preprocessing.LabelEncoder()
        df = pd.read_csv("sns dataset\sns{k}.csv".format(k=sns), index_col=0)
        df= df.drop(columns=['Urgency_Type'])
        df['Service'] = df['Service'].replace(df.Service.values, le.fit_transform(df.Service.values))
        df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
        train, test = train_test_split(df, test_size=0.25)
        
        train_y = train.Waiting_Time.values
        train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
        train_X = train_X.values
        test_y = test.Waiting_Time.values
        test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
        test_X = test_X.values
        train = train.drop(columns=["Waiting_Time","People_Waiting"])
        features_names = train.columns
    
    return train_X, train_y, test_X, test_y, features_names


#%%Run ElasticNet
def run_elasticNet():
       
    #train_X, train_y, test_X, test_y, features_names = getData("sns","1")
    #train_X, train_y, test_X, test_y, features_names = getData("sns_simple","1")
    #iterative_elasticnet(t_X=train_X, t_y=train_y, ts_X=test_X, ts_y=test_y, fs_ns=features_names,sns="1")
    #train_X, train_y, test_X, test_y, features_names = getData("sns_simple","2")
    #train_X, train_y, test_X, test_y, features_names = getData("sns","2")
    #iterative_elasticnet(t_X=train_X, t_y=train_y, ts_X=test_X, ts_y=test_y, fs_ns=features_names,sns="2")
    #train_X, train_y, test_X, test_y, features_names = getData("sns_simple","3")
    #train_X, train_y, test_X, test_y, features_names = getData("sns","3")
    #iterative_elasticnet(t_X=train_X, t_y=train_y, ts_X=test_X, ts_y=test_y, fs_ns=features_names,sns="3")
    #train_X, train_y, test_X, test_y, features_names = getData("sns_simple","4")
    #train_X, train_y, test_X, test_y, features_names = getData("sns","4")
    
    #train_X, train_y, test_X, test_y, features_names = getData("sns_simple2","1")
    #train_X, train_y, test_X, test_y, features_names = getData("sns_simple2","2")
    #train_X, train_y, test_X, test_y, features_names = getData("sns_simple2","3")
    train_X, train_y, test_X, test_y, features_names = getData("sns_simple2","4")
    iterative_elasticnet(t_X=train_X, t_y=train_y, ts_X=test_X, ts_y=test_y, fs_ns=features_names,sns="4")
    #elasticNet_CV(train_X,train_y,test_X,test_y,features_names)

import warnings
warnings.filterwarnings("ignore")

run_elasticNet()


# =============================================================================
# BOSTON DATASET
# reader = pd.read_csv("boston.csv", index_col=0)
# train, test = train_test_split(reader, test_size=0.2)
# 
# 
# 
# train_y = train.medv.values
# train_X = train.drop("medv", axis=1)
# features_names = train_X.columns
# train_X = train_X.values
# 
# test_y = test.medv.values
# test_X = test.drop("medv", axis=1)
# test_X = test_X.values
# =============================================================================

#SNS EMERGENCY LEVEL 1 DATASET

# =============================================================================
# df= pd.read_csv("sns dataset/sns1.csv", index_col=0)
# df= df.drop(columns=['Urgency_Type'])
# le = preprocessing.LabelEncoder()
# df['Service'] = df['Service'].replace(df.Service.values, le.fit_transform(df.Service.values))
# train, test = train_test_split(df, test_size=0.3)
# 
# train_y = train.Waiting_Time.values
# train_X = train.drop(columns=["Waiting_Time","People_Waiting"])
# train_X = train_X.values
# test_y = test.Waiting_Time.values
# test_X = test.drop(columns=["Waiting_Time","People_Waiting"])
# test_X = test_X.values
# train = train.drop(columns=["Waiting_Time","People_Waiting"])
# features_names = train.columns
# =============================================================================

        
        
