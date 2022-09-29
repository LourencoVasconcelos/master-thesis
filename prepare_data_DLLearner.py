# -*- coding: utf-8 -*-
"""
@author: Lourenço Vasconcelos
"""



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import myRuleGenerator as rf
import utils_prepare_data as u


#df tem o dataset inicial
def create_DlLearner_dataset(df,j,rules,sns,y=False):
    
    for rule in rules:
        for cond in rule.conditions:
            df[str(cond.conditionString())] = ""
            for row,col in df.iterrows():
                X = np.array(col)
                res = cond.get_label(X) # 1 ou 0
                df.at[row,str(cond.conditionString())]=res
                
            
    #new_df.to_csv('sns dataset/dataset_with_labels/sns1_labels.csv')
    s = False
    if 'Service' not in df.columns:
        df = df.drop(columns = ["Emergency_Stage","Weekday","Day","Month","Season","Yesterday_Waiting_Time"])
        s=True
    else:
        df = df.drop(columns = ["Emergency_Stage","Service","Weekday","Day","Month","Season","Yesterday_Waiting_Time"])
    df.index.name = 'sample_id'
    df.index += 1
    if s and y==False:
        df.to_csv('dataset_simple/sns{k}_ready{n}.csv'.format(n=j,k=sns))
    elif s==False and y==False:
        df.to_csv('dataset/sns{k}_ready{n}.csv'.format(n=j,k=sns))
    else:
        df.to_csv('dataset_simple_yesterday/sns{k}_ready{n}.csv'.format(n=j,k=sns))
        
    
    
    
def prepare_dataset(df,i,sns_,y=False):
    print("hey"+str(i))
    if 'Urgency_Type' in df.columns:
        df= df.drop(columns=['Urgency_Type'])
    le = preprocessing.LabelEncoder()
    s= True
    if 'Service' in df.columns:
        df['Service'] = df['Service'].replace(df.Service.values, le.fit_transform(df.Service.values))
        s = False
    df['Season'] = df['Season'].replace(df.Season.values, le.fit_transform(df.Season.values))
    train, test = train_test_split(df, test_size=0.3)
    
    train_y = train.Waiting_Time.values
    train = train.drop(columns=["Waiting_Time","People_Waiting"])
    test = test.drop(columns=["Waiting_Time","People_Waiting"])
    features_names = train.columns
    train_X = train
    train_X = train_X.values
    old_rules = rf.genRules(train_X, train_y, features_names, max_rules=2000)
    rules = rf.cleanRules(old_rules)
    n_df = df
    
    
    #df['Waiting_Time'] = df['Waiting_Time'].astype(int)
    #a=df['Waiting_Time'].unique()
    
    n_df = n_df.reset_index()
    n_df['Waiting_Time'] = n_df['Waiting_Time'].astype(int)
    new_df = n_df
    if s:
        if sns_=="1":
            new_df = u.add_v_simple(n_df,i)
        elif sns_=="2":
            new_df = u.add_v_simple2(n_df,i)
        elif sns_=="3":
            new_df = u.add_v_simple3(n_df,i)
        elif sns_=="4":
            new_df = u.add_v_simple4(n_df,i)
    else:
        if sns_ =="1":
            new_df = u.add_v(n_df,i)
        elif sns_ =="2":
            new_df = u.add_v_2(n_df,i)
        elif sns_ == "3":
            new_df = u.add_v_3(n_df,i)
        elif sns_ == "4":
            new_df = u.add_v_4(n_df,i)
    new_df = new_df.drop(columns=["Waiting_Time","People_Waiting"])
    
    create_DlLearner_dataset(new_df,i,rules,sns_,y) 


#creates a dataset ready for DL_Learner with rules as literals
def create_datasets_sns1(df):
    
    for i in [1,2,3,4,5]:
        prepare_dataset(df,i,"1")  
    df1 = df.query("Waiting_Time >=18")
    df1 = df1.query("Waiting_Time <=27")
    prepare_dataset(df1,6,"1") 
    df2 = df.query("Waiting_Time >=28")
    df2 = df2.query("Waiting_Time <=37")
    prepare_dataset(df2,7,"1") 
    df3 = df.query("Waiting_Time >=38")
    df3 = df3.query("Waiting_Time <=47.99")
    prepare_dataset(df3,8,"1") 
    df4 = df.query("Waiting_Time >=48")
    df4 = df4.query("Waiting_Time <=57.99")
    prepare_dataset(df4,9,"1") 
    df5 = df.query("Waiting_Time >=58")
    df5 = df5.query("Waiting_Time <=67.99")
    prepare_dataset(df5,10,"1") 
    df6 = df.query("Waiting_Time >=68")
    df6 = df6.query("Waiting_Time <=77.99")
    prepare_dataset(df6,11,"1") 
    df7 = df.query("Waiting_Time >=78")
    df7 = df7.query("Waiting_Time <=87.99")
    prepare_dataset(df7,12,"1") 
    df8 = df.query("Waiting_Time >=88")
    df8 = df8.query("Waiting_Time <=97.99")
    prepare_dataset(df8,13,"1") 
    df9 = df.query("Waiting_Time >=108")
    df9 = df9.query("Waiting_Time <=117.99")
    prepare_dataset(df9,14,"1") 
    df10 = df.query("Waiting_Time >=118")
    df10 = df10.query("Waiting_Time <=127.99")
    prepare_dataset(df10,15,"1") 
    df11 = df.query("Waiting_Time >=128")
    df11 = df11.query("Waiting_Time <=137.99")
    prepare_dataset(df11,16,"1") 
    df12 = df.query("Waiting_Time >=138")
    df12 = df12.query("Waiting_Time <=147.99")
    prepare_dataset(df12,17,"1") 
    df13 = df.query("Waiting_Time >=148")
    df13 = df13.query("Waiting_Time <=157.99")
    prepare_dataset(df13,18,"1") 
    df14 = df.query("Waiting_Time >=158")
    df14 = df14.query("Waiting_Time <=167.99")
    prepare_dataset(df14,19,"1") 
    df15 = df.query("Waiting_Time >=168")
    df15 = df15.query("Waiting_Time <=177.99")
    prepare_dataset(df15,20,"1") 

def create_datasets_sns1_simple(df,y=False):
    
    for i in [1,2,3,4,5]:
        prepare_dataset(df,i,"1",y)  
    df1 = df.query("Waiting_Time >=73")
    df1 = df1.query("Waiting_Time <=82.99")
    prepare_dataset(df1,6,"1",y) 
    df2 = df.query("Waiting_Time >=83")
    df2 = df2.query("Waiting_Time <=92.99")
    prepare_dataset(df2,7,"1",y) 
    df3 = df.query("Waiting_Time >=99")
    df3 = df3.query("Waiting_Time <=108.99")
    prepare_dataset(df3,8,"1",y) 
    df4 = df.query("Waiting_Time >=109")
    df4 = df4.query("Waiting_Time <=118.99")
    prepare_dataset(df4,9,"1",y) 
    df5 = df.query("Waiting_Time >=119")
    df5 = df5.query("Waiting_Time <=128.99")
    prepare_dataset(df5,10,"1",y) 
    df6 = df.query("Waiting_Time >=135")
    df6 = df6.query("Waiting_Time <=144.99")
    prepare_dataset(df6,11,"1",y) 
    df7 = df.query("Waiting_Time >=145")
    df7 = df7.query("Waiting_Time <=154.99")
    prepare_dataset(df7,12,"1",y) 
    df8 = df.query("Waiting_Time >=155")
    df8 = df8.query("Waiting_Time <=164.99")
    prepare_dataset(df8,13,"1",y) 
    df9 = df.query("Waiting_Time >=75")
    df9 = df9.query("Waiting_Time <=84.99")
    prepare_dataset(df9,14,"1",y) 
    df10 = df.query("Waiting_Time >=79")
    df10 = df10.query("Waiting_Time <=88.99")
    prepare_dataset(df10,15,"1",y) 
    df11 = df.query("Waiting_Time >=85")
    df11 = df11.query("Waiting_Time <=94.99")
    prepare_dataset(df11,16,"1",y) 
    df12 = df.query("Waiting_Time >=105")
    df12 = df12.query("Waiting_Time <=114.99")
    prepare_dataset(df12,17,"1",y) 
    df13 = df.query("Waiting_Time >=115")
    df13 = df13.query("Waiting_Time <=124.99")
    prepare_dataset(df13,18,"1",y) 
    df14 = df.query("Waiting_Time >=140")
    df14 = df14.query("Waiting_Time <=149.99")
    prepare_dataset(df14,19,"1",y) 
    df15 = df.query("Waiting_Time >=150")
    df15 = df15.query("Waiting_Time <=159.99")
    prepare_dataset(df15,20,"1",y) 
    
    
#%% SNS2

def create_datasets_sns2_simple(df,y=False):
    
    for i in [1,2,3,4,5]:
        prepare_dataset(df,i,"2",y)  
    df1 = df.query("Waiting_Time >=45")
    df1 = df1.query("Waiting_Time <=54.99")
    prepare_dataset(df1,6,"2",y) 
    df2 = df.query("Waiting_Time >=55")
    df2 = df2.query("Waiting_Time <=64.99")
    prepare_dataset(df2,7,"2",y) 
    df3 = df.query("Waiting_Time >=65")
    df3 = df3.query("Waiting_Time <=74.99")
    prepare_dataset(df3,8,"2",y) 
    df4 = df.query("Waiting_Time >=75")
    df4 = df4.query("Waiting_Time <=84.99")
    prepare_dataset(df4,9,"2",y) 
    df5 = df.query("Waiting_Time >=85")
    df5 = df5.query("Waiting_Time <=94.99")
    prepare_dataset(df5,10,"2",y) 
    df6 = df.query("Waiting_Time >=95")
    df6 = df6.query("Waiting_Time <=104.99")
    prepare_dataset(df6,11,"2",y) 
    df7 = df.query("Waiting_Time >=105")
    df7 = df7.query("Waiting_Time <=114.99")
    prepare_dataset(df7,12,"2",y) 
    df8 = df.query("Waiting_Time >=113")
    df8 = df8.query("Waiting_Time <=122.99")
    prepare_dataset(df8,13,"2",y) 
    df9 = df.query("Waiting_Time >=50")
    df9 = df9.query("Waiting_Time <=59.99")
    prepare_dataset(df9,14,"2",y) 
    df10 = df.query("Waiting_Time >=60")
    df10 = df10.query("Waiting_Time <=69.99")
    prepare_dataset(df10,15,"2",y) 
    df11 = df.query("Waiting_Time >=70")
    df11 = df11.query("Waiting_Time <=79.99")
    prepare_dataset(df11,16,"2",y) 
    df12 = df.query("Waiting_Time >=80")
    df12 = df12.query("Waiting_Time <=89.99")
    prepare_dataset(df12,17,"2",y) 
    df13 = df.query("Waiting_Time >=90")
    df13 = df13.query("Waiting_Time <=99.99")
    prepare_dataset(df13,18,"2",y) 
    df14 = df.query("Waiting_Time >=100")
    df14 = df14.query("Waiting_Time <=109.99")
    prepare_dataset(df14,19,"2",y) 
    df15 = df.query("Waiting_Time >=110")
    df15 = df15.query("Waiting_Time <=119.99")
    prepare_dataset(df15,20,"2",y) 
    

def create_datasets_sns2(df):
    
    for i in [1,2,3,4,5]:
        prepare_dataset(df,i,"2")  
    df1 = df.query("Waiting_Time >=28")
    df1 = df1.query("Waiting_Time <=37.99")
    prepare_dataset(df1,6,"2") 
    df2 = df.query("Waiting_Time >=38")
    df2 = df2.query("Waiting_Time <=47.99")
    prepare_dataset(df2,7,"2") 
    df3 = df.query("Waiting_Time >=48")
    df3 = df3.query("Waiting_Time <=57.99")
    prepare_dataset(df3,8,"2") 
    df4 = df.query("Waiting_Time >=58")
    df4 = df4.query("Waiting_Time <=67.99")
    prepare_dataset(df4,9,"2") 
    df5 = df.query("Waiting_Time >=68")
    df5 = df5.query("Waiting_Time <=77.99")
    prepare_dataset(df5,10,"2") 
    df6 = df.query("Waiting_Time >=78")
    df6 = df6.query("Waiting_Time <=87.99")
    prepare_dataset(df6,11,"2") 
    df7 = df.query("Waiting_Time >=88")
    df7 = df7.query("Waiting_Time <=97.99")
    prepare_dataset(df7,12,"2") 
    df8 = df.query("Waiting_Time >=98")
    df8 = df8.query("Waiting_Time <=107.99")
    prepare_dataset(df8,13,"2") 
    df9 = df.query("Waiting_Time >=107")
    df9 = df9.query("Waiting_Time <=117.99")
    prepare_dataset(df9,14,"2") 
    df10 = df.query("Waiting_Time >=118")
    df10 = df10.query("Waiting_Time <=127.99")
    prepare_dataset(df10,15,"2") 
    df11 = df.query("Waiting_Time >=139")
    df11 = df11.query("Waiting_Time <=148.99")
    prepare_dataset(df11,16,"2") 
    df12 = df.query("Waiting_Time >=43")
    df12 = df12.query("Waiting_Time <=52.99")
    prepare_dataset(df12,17,"2") 
    df13 = df.query("Waiting_Time >=63")
    df13 = df13.query("Waiting_Time <=72.99")
    prepare_dataset(df13,18,"2") 
    df14 = df.query("Waiting_Time >=73")
    df14 = df14.query("Waiting_Time <=82.99")
    prepare_dataset(df14,19,"2") 
    df15 = df.query("Waiting_Time >=83")
    df15 = df15.query("Waiting_Time <=92.99")
    prepare_dataset(df15,20,"2") 
    
    
    
#%% SNS 3


def create_datasets_sns3_simple(df,y=False):
    
    for i in [1,2,3,4,5]:
        prepare_dataset(df,i,"3",y)  
    df1 = df.query("Waiting_Time >=29")
    df1 = df1.query("Waiting_Time <=38.99")
    prepare_dataset(df1,6,"3",y) 
    df2 = df.query("Waiting_Time >=39")
    df2 = df2.query("Waiting_Time <=48.99")
    prepare_dataset(df2,7,"3",y) 
    df3 = df.query("Waiting_Time >=49")
    df3 = df3.query("Waiting_Time <=58.99")
    prepare_dataset(df3,8,"3",y) 
    df4 = df.query("Waiting_Time >=59")
    df4 = df4.query("Waiting_Time <=68.99")
    prepare_dataset(df4,9,"3",y) 
    df5 = df.query("Waiting_Time >=69")
    df5 = df5.query("Waiting_Time <=78.99")
    prepare_dataset(df5,10,"3",y) 
    df6 = df.query("Waiting_Time >=79")
    df6 = df6.query("Waiting_Time <=88.99")
    prepare_dataset(df6,11,"3",y) 
    df7 = df.query("Waiting_Time >=89")
    df7 = df7.query("Waiting_Time <=98.99")
    prepare_dataset(df7,12,"3",y) 
    df8 = df.query("Waiting_Time >=32")
    df8 = df8.query("Waiting_Time <=41.99")
    prepare_dataset(df8,13,"3",y) 
    df9 = df.query("Waiting_Time >=42")
    df9 = df9.query("Waiting_Time <=51.99")
    prepare_dataset(df9,14,"3",y) 
    df10 = df.query("Waiting_Time >=52")
    df10 = df10.query("Waiting_Time <=61.99")
    prepare_dataset(df10,15,"3",y) 
    df11 = df.query("Waiting_Time >=62")
    df11 = df11.query("Waiting_Time <=71.99")
    prepare_dataset(df11,16,"3",y) 
    df12 = df.query("Waiting_Time >=72")
    df12 = df12.query("Waiting_Time <=81.99")
    prepare_dataset(df12,17,"3",y) 
    df13 = df.query("Waiting_Time >=82")
    df13 = df13.query("Waiting_Time <=91.99")
    prepare_dataset(df13,18,"3",y) 
    df14 = df.query("Waiting_Time >=92")
    df14 = df14.query("Waiting_Time <=102.99")
    prepare_dataset(df14,19,"3",y) 
    df15 = df.query("Waiting_Time >=56")
    df15 = df15.query("Waiting_Time <=65.99")
    prepare_dataset(df15,20,"3",y) 
    
def create_datasets_sns3(df):
    
    for i in [1,2,3,4,5]:
        prepare_dataset(df,i,"3")  
    df1 = df.query("Waiting_Time >=22")
    df1 = df1.query("Waiting_Time <=31.99")
    prepare_dataset(df1,6,"3") 
    df2 = df.query("Waiting_Time >=32")
    df2 = df2.query("Waiting_Time <=41.99")
    prepare_dataset(df2,7,"3") 
    df3 = df.query("Waiting_Time >=42")
    df3 = df3.query("Waiting_Time <=51.99")
    prepare_dataset(df3,8,"3") 
    df4 = df.query("Waiting_Time >=52")
    df4 = df4.query("Waiting_Time <=61.99")
    prepare_dataset(df4,9,"3") 
    df5 = df.query("Waiting_Time >=62")
    df5 = df5.query("Waiting_Time <=71.99")
    prepare_dataset(df5,10,"3") 
    df6 = df.query("Waiting_Time >=72")
    df6 = df6.query("Waiting_Time <=81.99")
    prepare_dataset(df6,11,"3") 
    df7 = df.query("Waiting_Time >=82")
    df7 = df7.query("Waiting_Time <=91.99")
    prepare_dataset(df7,12,"3") 
    df8 = df.query("Waiting_Time >=92")
    df8 = df8.query("Waiting_Time <=101.99")
    prepare_dataset(df8,13,"3") 
    df9 = df.query("Waiting_Time >=102")
    df9 = df9.query("Waiting_Time <=111.99")
    prepare_dataset(df9,14,"3") 
    df10 = df.query("Waiting_Time >=137")
    df10 = df10.query("Waiting_Time <=146.99")
    prepare_dataset(df10,15,"3") 
    df11 = df.query("Waiting_Time >=47")
    df11 = df11.query("Waiting_Time <=56.99")
    prepare_dataset(df11,16,"3") 
    df12 = df.query("Waiting_Time >=57")
    df12 = df12.query("Waiting_Time <=66.99")
    prepare_dataset(df12,17,"3") 
    df13 = df.query("Waiting_Time >=67")
    df13 = df13.query("Waiting_Time <=76.99")
    prepare_dataset(df13,18,"3") 
    df14 = df.query("Waiting_Time >=87")
    df14 = df14.query("Waiting_Time <=96.99")
    prepare_dataset(df14,19,"3") 
    df15 = df.query("Waiting_Time >=97")
    df15 = df15.query("Waiting_Time <=107.99")
    prepare_dataset(df15,20,"3")     
    
    
#%% SNS4

def create_datasets_sns4_simple(df,y=False):
    
    for i in [1,2,3,4,5]:
        prepare_dataset(df,i,"4",y)  
    df1 = df.query("Waiting_Time >=8")
    df1 = df1.query("Waiting_Time <=17.99")
    prepare_dataset(df1,6,"4",y) 
    df2 = df.query("Waiting_Time >=18")
    df2 = df2.query("Waiting_Time <=27.99")
    prepare_dataset(df2,7,"4",y) 
    df3 = df.query("Waiting_Time >=28")
    df3 = df3.query("Waiting_Time <=37.99")
    prepare_dataset(df3,8,"4",y) 
    df4 = df.query("Waiting_Time >=37")
    df4 = df4.query("Waiting_Time <=46.99")
    prepare_dataset(df4,9,"4",y) 
    df5 = df.query("Waiting_Time >=10")
    df5 = df5.query("Waiting_Time <=19.99")
    prepare_dataset(df5,10,"4",y) 
    df6 = df.query("Waiting_Time >=20")
    df6 = df6.query("Waiting_Time <=29.99")
    prepare_dataset(df6,11,"4",y) 
    df7 = df.query("Waiting_Time >=30")
    df7 = df7.query("Waiting_Time <=39.99")
    prepare_dataset(df7,12,"4",y) 
    df8 = df.query("Waiting_Time >=12")
    df8 = df8.query("Waiting_Time <=21.99")
    prepare_dataset(df8,13,"4",y) 
    df9 = df.query("Waiting_Time >=22")
    df9 = df9.query("Waiting_Time <=31.99")
    prepare_dataset(df9,14,"4",y) 
    df10 = df.query("Waiting_Time >=32")
    df10 = df10.query("Waiting_Time <=41.99")
    prepare_dataset(df10,15,"4",y) 
    df11 = df.query("Waiting_Time >=14")
    df11 = df11.query("Waiting_Time <=23.99")
    prepare_dataset(df11,16,"4",y) 
    df12 = df.query("Waiting_Time >=24")
    df12 = df12.query("Waiting_Time <=33.99")
    prepare_dataset(df12,17,"4",y) 
    df13 = df.query("Waiting_Time >=34")
    df13 = df13.query("Waiting_Time <=43.99")
    prepare_dataset(df13,18,"4",y) 
    df14 = df.query("Waiting_Time >=16")
    df14 = df14.query("Waiting_Time <=25.99")
    prepare_dataset(df14,19,"4",y) 
    df15 = df.query("Waiting_Time >=26")
    df15 = df15.query("Waiting_Time <=35.99")
    prepare_dataset(df15,20,"4",y) 
    
def create_datasets_sns4(df):
    
    for i in [1,2,3,4,5]:
        prepare_dataset(df,i,"4")  
    df1 = df.query("Waiting_Time >=4")
    df1 = df1.query("Waiting_Time <=13.99")
    prepare_dataset(df1,6,"4") 
    df2 = df.query("Waiting_Time >=14")
    df2 = df2.query("Waiting_Time <=23.99")
    prepare_dataset(df2,7,"4") 
    df3 = df.query("Waiting_Time >=24")
    df3 = df3.query("Waiting_Time <=33.99")
    prepare_dataset(df3,8,"4") 
    df4 = df.query("Waiting_Time >=34")
    df4 = df4.query("Waiting_Time <=43.99")
    prepare_dataset(df4,9,"4") 
    df5 = df.query("Waiting_Time >=44")
    df5 = df5.query("Waiting_Time <=53.99")
    prepare_dataset(df5,10,"4") 
    df6 = df.query("Waiting_Time >=45")
    df6 = df6.query("Waiting_Time <=54.99")
    prepare_dataset(df6,11,"4") 
    df7 = df.query("Waiting_Time >=46")
    df7 = df7.query("Waiting_Time <=55.99")
    prepare_dataset(df7,12,"4") 
    df8 = df.query("Waiting_Time >=47")
    df8 = df8.query("Waiting_Time <=56.99")
    prepare_dataset(df8,13,"4") 
    df9 = df.query("Waiting_Time >=6")
    df9 = df9.query("Waiting_Time <=15.99")
    prepare_dataset(df9,14,"4") 
    df10 = df.query("Waiting_Time >=16")
    df10 = df10.query("Waiting_Time <=25.99")
    prepare_dataset(df10,15,"4") 
    df11 = df.query("Waiting_Time >=26")
    df11 = df11.query("Waiting_Time <=35.99")
    prepare_dataset(df11,16,"4") 
    df12 = df.query("Waiting_Time >=36")
    df12 = df12.query("Waiting_Time <=45.99")
    prepare_dataset(df12,17,"4") 
    df13 = df.query("Waiting_Time >=8")
    df13 = df13.query("Waiting_Time <=17.99")
    prepare_dataset(df13,18,"4") 
    df14 = df.query("Waiting_Time >=18")
    df14 = df14.query("Waiting_Time <=27.99")
    prepare_dataset(df14,19,"4") 
    df15 = df.query("Waiting_Time >=28")
    df15 = df15.query("Waiting_Time <=37.99")
    prepare_dataset(df15,20,"4") 
    