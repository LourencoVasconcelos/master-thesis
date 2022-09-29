# -*- coding: utf-8 -*-
"""
@author: Lourenço Vasconcelos
"""

import numpy as np

def add_v(n_df,i):
    n_df['target'] = ""
    
    if i<=5:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']/40)
    elif i==6:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-17)
    elif i==7:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-27)
    elif i==8:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-37)
    elif i==9:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-47)
    elif i==10:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-57)
    elif i==11:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-67)
    elif i==12:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-77)
    elif i==13:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-87)
    elif i==14:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-107)
    elif i==15:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-117)
    elif i==16:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-127)
    elif i==17:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-137)
    elif i==18:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-147)
    elif i==19:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-157)
    elif i==20:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-167)
            
    return n_df


def add_v_simple(n_df,i):
    n_df['target'] = ""
    
    if i<=5:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']/35)
    elif i==6:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-72)
    elif i==7:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-82)
    elif i==8:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-98)
    elif i==9:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-108)
    elif i==10:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-118)
    elif i==11:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-134)
    elif i==12:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-144)
    elif i==13:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-154)
    elif i==14:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-74)
    elif i==15:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-78)
    elif i==16:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-84)
    elif i==17:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-104)
    elif i==18:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-114)
    elif i==19:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-139)
    elif i==20:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-149)
            
    return n_df

#%% SNS2

def add_v_simple2(n_df,i):
    n_df['target'] = ""
    
    if i<=5:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']/17)
    elif i==6:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-44)
    elif i==7:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-54)
    elif i==8:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-64)
    elif i==9:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-74)
    elif i==10:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-84)
    elif i==11:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-94)
    elif i==12:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-104)
    elif i==13:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-112)
    elif i==14:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-49)
    elif i==15:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-59)
    elif i==16:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-69)
    elif i==17:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-79)
    elif i==18:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-89)
    elif i==19:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-99)
    elif i==20:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-109)
            
    return n_df



def add_v_2(n_df,i):
    n_df['target'] = ""
    
    if i<=5:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(np.ceil(col['Waiting_Time']/27))
    elif i==6:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-27)
    elif i==7:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-37)
    elif i==8:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-47)
    elif i==9:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-57)
    elif i==10:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-67)
    elif i==11:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-77)
    elif i==12:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-87)
    elif i==13:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-97)
    elif i==14:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-107)
    elif i==15:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-117)
    elif i==16:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-138)
    elif i==17:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-42)
    elif i==18:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-62)
    elif i==19:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-72)
    elif i==20:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-82)
            
    return n_df


#%% SNS 3

def add_v_simple3(n_df,i):
    n_df['target'] = ""
    
    if i<=5:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(round(col['Waiting_Time']/18))
    elif i==6:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-28)
    elif i==7:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-38)
    elif i==8:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-48)
    elif i==9:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-58)
    elif i==10:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-68)
    elif i==11:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-78)
    elif i==12:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-88)
    elif i==13:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-31)
    elif i==14:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-41)
    elif i==15:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-51)
    elif i==16:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-61)
    elif i==17:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-71)
    elif i==18:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-81)
    elif i==19:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-91)
    elif i==20:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-55)
            
    return n_df



def add_v_3(n_df,i):
    n_df['target'] = ""
    
    if i<=5:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(np.round(col['Waiting_Time']/24))
    elif i==6:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-21)
    elif i==7:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-31)
    elif i==8:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-41)
    elif i==9:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-51)
    elif i==10:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-61)
    elif i==11:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-71)
    elif i==12:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-81)
    elif i==13:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-91)
    elif i==14:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-101)
    elif i==15:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-136)
    elif i==16:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-46)
    elif i==17:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-56)
    elif i==18:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-66)
    elif i==19:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-86)
    elif i==20:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-96)
            
    return n_df

#%%% SNS 4


def add_v_simple4(n_df,i):
    n_df['target'] = ""
    
    if i<=5:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']/7.5)
    elif i==6:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-7)
    elif i==7:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-17)
    elif i==8:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-27)
    elif i==9:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-36)
    elif i==10:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-9)
    elif i==11:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-19)
    elif i==12:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-29)
    elif i==13:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-11)
    elif i==14:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-21)
    elif i==15:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-31)
    elif i==16:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-13)
    elif i==17:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-23)
    elif i==18:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-33)
    elif i==19:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-15)
    elif i==20:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-25)
            
    return n_df



def add_v_4(n_df,i):
    n_df['target'] = ""
    
    if i<=5:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(np.ceil(col['Waiting_Time']/14))
    elif i==6:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-3)
    elif i==7:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-13)
    elif i==8:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-23)
    elif i==9:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-33)
    elif i==10:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-43)
    elif i==11:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-44)
    elif i==12:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-45)
    elif i==13:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-46)
    elif i==14:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-5)
    elif i==15:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-15)
    elif i==16:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-25)
    elif i==17:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-35)
    elif i==18:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-7)
    elif i==19:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-17)
    elif i==20:
        for row,col in n_df.iterrows():
            n_df.at[row,'target'] = int(col['Waiting_Time']-27)
            
    return n_df



