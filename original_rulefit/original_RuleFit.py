# -*- coding: utf-8 -*-
"""
@author: Lourenço Vasconcelos
"""

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn.metrics as sm
import pandas as pd
from rulefit import RuleFit

#%%Dataset with Service columns
# le = preprocessing.LabelEncoder()
# df = pd.read_csv("sns dataset\sns1.csv", index_col=0)
# df= df.drop(columns=['Urgency_Type'])
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

#%%Dataset without Service column
le = preprocessing.LabelEncoder()
#df = pd.read_csv("sns dataset\sns1_simple2.csv", index_col=0)
#df = pd.read_csv("sns dataset\sns2_simple2.csv", index_col=0)
#df = pd.read_csv("sns dataset\sns3_simple2.csv", index_col=0)
df = pd.read_csv("sns dataset\sns4_simple2.csv", index_col=0)
#df = pd.read_csv("sns dataset\sns1_simple.csv", index_col=0)
#df = pd.read_csv("sns dataset\sns2_simple.csv", index_col=0)
#df = pd.read_csv("sns dataset\sns2.csv", index_col=0)
#df = pd.read_csv("sns dataset\sns3_simple.csv", index_col=0)
#df = pd.read_csv("sns dataset\sns3.csv", index_col=0)
#df = pd.read_csv("sns dataset\sns4_simple.csv", index_col=0)
#df = pd.read_csv("sns dataset\sns4.csv", index_col=0)
df=df.set_index('Emergency_Stage') #for sns_simple
#df= df.drop(columns=['Urgency_Type']) #For sns with service column
#df['Service'] = df['Service'].replace(df.Service.values, le.fit_transform(df.Service.values)) #For sns with service column
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



rf = RuleFit(tree_size=4, sample_fract='default', max_rules=2000,
             memory_par=0.01, tree_generator=None,
             rfmode='regress', lin_trim_quantile=0.025,
             lin_standardise=True, exp_rand_tree_size=True, random_state=1) 
rf.fit(train_X, train_y, feature_names=features_names)
y_pred_t = rf.predict(train_X)
y_pred_v = rf.predict(test_X)

#2000, 1000, 500, 200, 100, 50

rules = rf.get_rules()
rules = rules[rules.coef != 0].sort_values(by="support",ascending=False)
num_rules = len(rules[rules.type == 'rule'])
print(rules)
print("Mean Absolute Error Train: " + str(round(sm.mean_absolute_error(train_y, y_pred_t), 2)))
print("Mean Absolute Error Test: " + str(round(sm.mean_absolute_error(test_y, y_pred_v), 2)))
print("Number Of Rules:" + str(num_rules))

print("Mean Squared Error Train: "+ str(sm.mean_squared_error(train_y, y_pred_t)))
print("Mean Squared Error Test: "+ str(sm.mean_squared_error(test_y, y_pred_v)))
