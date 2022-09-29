# -*- coding: utf-8 -*-
"""
@author: Lourenço Vasconcelos
"""

import myRuleGenerator as rf
import sklearn.metrics as sm
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, Ridge, RidgeCV
import matplotlib.pyplot as plt


    
#old
def elasticNet_afterDL(rules, train_X, train_y, test_X, test_y):
    
    X_concat,_,_,_,_ = rf.prepareData(rules, train_X, trim_quantile = 0.025) # alphaL = 10**i,
    regr = ElasticNetCV(random_state=0,  tol=0.0005, max_iter=5000) #alpha = alphaL,
    regr.fit(X_concat, train_y)
    
    train_predictions, rules_length = rf.GetPredictions(regr, train_X, rules)
    test_predictions, rules_length = rf.GetPredictions(regr, test_X, rules)
    print(round(sm.mean_absolute_error(train_y, train_predictions), 2))
    print(round(sm.mean_absolute_error(test_y, test_predictions), 2))
    print(rules_length)
    
#old
def lasso_afterDL(rules, train_X, train_y, test_X, test_y):
    
    X_concat,_,_,_,_ = rf.prepareData(rules, train_X, trim_quantile = 0.025) # alphaL = 10**i,
    regr = LassoCV(random_state=0,  tol=0.0005, max_iter=5000) #alpha = alphaL,
    regr.fit(X_concat, train_y)
    
    train_predictions, rules_length = rf.GetPredictions(regr, train_X, rules)
    test_predictions, rules_length = rf.GetPredictions(regr, test_X, rules)
    print(round(sm.mean_absolute_error(train_y, train_predictions), 2))
    print(round(sm.mean_absolute_error(test_y, test_predictions), 2))
    print(rules_length)
    
#old
def ridge_afterDL(rules, train_X, train_y, test_X, test_y):
    
    X_concat,_,_,_,_ = rf.prepareData(rules, train_X, trim_quantile = 0.025) # alphaL = 10**i,
    regr = RidgeCV() #alpha = alphaL,
    regr.fit(X_concat, train_y)
    
    train_predictions, rules_length = rf.GetPredictions(regr, train_X, rules)
    test_predictions, rules_length = rf.GetPredictions(regr, test_X, rules)
    print(round(sm.mean_absolute_error(train_y, train_predictions), 2))
    print(round(sm.mean_absolute_error(test_y, test_predictions), 2))
    print(rules_length)


#old
def Lasso_model(train_X, train_y, rules, test_X, test_y):
    rules = rf.cleanRules(rules)

    mean = train_X.mean()
    stdev = train_X.std()
    train_X = (train_X-mean)/stdev
    test_X = (test_X-mean)/stdev
    
    #scaler = StandardScaler()
    #f = scaler.fit(train_X)
    #train_X = f.transform(train_X)
    #test_X = f.transform(test_X)
    
    
    ls = Lasso(alpha=10**0, tol=0.0005, max_iter=5000)
    #usar Lasso sem a cv, 
    
    ls.fit(train_X, train_y)
    train_predictions = ls.predict(train_X)
    test_predictions = ls.predict(test_X)
     
    #train_predictions, rules_length,_,_,_ = rf.GetPredictions(ls, train_X, rules)
    #test_predictions, rules_length,_,_,_ = rf.GetPredictions(ls, test_X, rules)
        
    print("Train Error: " + str(round(sm.mean_absolute_error(train_y, train_predictions), 2)))
    print("Validation error: "+ str(round(sm.mean_absolute_error(test_y, test_predictions), 2)))
    print(ls.coef_)
    
    
    
    
#Make plots for different models after running the model after dl_learner
def make_plots(l1r, rules_length,ridge_values,val_MAE,train_MAE,sns):
    if l1r == 0.5 or l1r ==1:
        print(rules_length)
    else:
        print(ridge_values)
        print(rules_length)
    plt.title("Accuracy by number of rules")
    plt.ylabel("Number of relevant rules")
    plt.xlabel("Mean Absolute Error")
    if l1r==0.5 or l1r==1:
        plt.plot(val_MAE,rules_length, 'r')
        plt.plot(train_MAE,rules_length, 'r--')
    else:
        plt.ylabel("Alpha value")
        plt.plot(val_MAE,ridge_values, 'r')
        plt.plot(train_MAE,ridge_values, 'r--')        
        plt.yscale("linear")
    #plt.xscale('symlog')
    plt.legend(["Validation", "Train"], loc="upper right")
    if l1r==0.5:
        plt.savefig("plots_results_sns{k}/Accuracy_DL_ElasticNet.pdf".format(k=sns), format="pdf",bbox_inches="tight")
    if l1r==1:
        plt.savefig("plots_results_sns{k}/Accuracy_DL_Lasso.pdf".format(k=sns), format="pdf",bbox_inches="tight")
    if l1r==0:
        plt.savefig("plots_results_sns{k}/Accuracy_DL_Ridge.pdf".format(k=sns), format="pdf",bbox_inches="tight")
        plt.title("Accuracy by Alpha value")
    plt.show()
    
    
    
def model_afterDL(rules, train_X, train_y, test_X, test_y, l1r,sns_):
    train_MAE = []
    val_MAE = []
    rules_length = []
    X_concat,_,_,_,_ = rf.prepareData(rules, train_X)
    ridge_values = [10**-5,10**-1,10**1,10**1.2,10**1.3,10**1.4,10**1.5,10**2]
    if l1r==0.5:
        #for i in (-0.5,0,0.5,0.7,0.8,1): #SNS 1, with and without + SNS 2 without
        #for i in (-0.5,0,0.3,0.5,0.6,0.7):  #SNS2 with service column
        #for i in (-0.5,-0.2,0,0.2,0.4,0.6):  #SNS3 with service column
        #for i in (-0.5,-0.3,-0.1,0,0.05,0.1,0.15):  #SNS4 without service column
        #for i in (-0.5,-0.4,-0.2,0,0.1,0.2,0.25):  #SNS4 with service column & SNS2 with yesterday
        #for i in (-0.5,-0.4,-0.2,0,0.2,0.4,0.5):  #SNS3 with yesterday
        #for i in (-1,-0.8,-0.6,-0.4,-0.2,0,0.1):  #SNS4 with yesterday
        #for i in (-0.1,0,0.2,0.4,0.6,0.8):  #SNS1 with yesterday
        for i in (-0.1,0,0.2,0.4,0.6,0.8):  #SNS2 with yesterday
            en = ElasticNet(alpha = 10**i, random_state=0,  tol=0.0005, max_iter=5000) #
            en.fit(X_concat, train_y)
            train_predictions, number_rules = rf.GetPredictions(en, train_X, rules)
            test_predictions, number_rules = rf.GetPredictions(en, test_X, rules)
            train_MAE.append(round(sm.mean_absolute_error(train_y, train_predictions), 2))
            val_MAE.append(round(sm.mean_absolute_error(test_y, test_predictions), 2))
            rules_length.append(number_rules)
    if l1r==1:
       #for i in (-0.5,-0.2,0,0.3,0.5,0.7):#SNS 1, with and without + SNS 2 without
        #for i in (-0.5,-0.2,-0.1,0,0.1,0.3):  #SNS2 with service column
        #for i in (-0.5,-0.45,-0.4,-0.2,0,0.2):  #SNS3 with service column
        #for i in (-1,-0.7,-0.5,-0.3,-0.2,-0.05):  #SNS4 without service column
        #for i in (-1,-0.8,-0.6,-0.4,-0.15,-0.05):  #SNS4 with service column & SNS2 with yesterday
        #for i in (-0.7,-0.3,-0.2,0.2,0.4,0.6):  #SNS1 with yesterday
        #for i in (-0.7,-0.5,-0.3,-0.1,0.1,0.3):  #SNS3 with yesterday
        for i in (-1,-0.8,-0.6,-0.5,-0.3,-0.25):  #SNS4 with yesterday
            ls = Lasso(alpha = 10**i, random_state=0,  tol=0.0005, max_iter=5000) #
            ls.fit(X_concat, train_y) 
            train_predictions, number_rules = rf.GetPredictions(ls, train_X, rules)
            test_predictions, number_rules = rf.GetPredictions(ls, test_X, rules)
            train_MAE.append(round(sm.mean_absolute_error(train_y, train_predictions), 2))
            val_MAE.append(round(sm.mean_absolute_error(test_y, test_predictions), 2))
            rules_length.append(number_rules)
    if l1r==0: #Pointless, Ridge Regression never puts coefs at 0 (square in formula), just to check accuracy changes
        for i in (ridge_values):
            regr= Ridge(alpha = i, max_iter=5000)
            regr.fit(X_concat,train_y)
            train_predictions, number_rules = rf.GetPredictions(regr, train_X, rules)
            test_predictions, number_rules = rf.GetPredictions(regr, test_X, rules)
            train_MAE.append(round(sm.mean_absolute_error(train_y, train_predictions), 2))
            val_MAE.append(round(sm.mean_absolute_error(test_y, test_predictions), 2))
            rules_length.append(number_rules)
            
    print(train_MAE)
    print(val_MAE)
    make_plots(l1r, rules_length,ridge_values,val_MAE,train_MAE,sns_)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#%%##OLD
