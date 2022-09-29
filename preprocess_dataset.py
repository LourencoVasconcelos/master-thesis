# -*- coding: utf-8 -*-
"""
@author: Lourenço Vasconcelos
"""


import pandas as pd
import pre_process_aux as aux

# %% Prepare Data for Rule based Methods
df = pd.read_csv("sns_df.csv", index_col=0)

df = df.query("H_Name == 'Santa Maria'")


df['Acquisition_Time'] = pd.to_datetime(df['Acquisition_Time'])

df = df.query("Service == 'Cirurgia' | Service == 'Medicina'" )


df['Acquisition_Time'] = df.Acquisition_Time.map(lambda x: x.strftime('%Y-%m-%d')) #-%H
df['Acquisition_Time'] = pd.to_datetime(df['Acquisition_Time'])


df1 = df.query("Emergency_Stage == 1")
df2 = df.query("Emergency_Stage == 2")
df3 = df.query("Emergency_Stage == 3")
df4 = df.query("Emergency_Stage == 4")



###############  OUTLIERS EMERGENCY LEVEL 1
mean1 = df1.groupby("Emergency_Stage", as_index=False).mean()['Waiting_Time'][0]
std1 = df1.groupby("Emergency_Stage",as_index=False).std()['Waiting_Time'][0]

cut_off1 = std1*3
lower1, upper1 = 0, mean1 + cut_off1 #mean-cut_off would be <0

# identify outliers df1
outliers1 = [x for x in df1['Waiting_Time'] if x < lower1 or x > upper1]
#df1 = df1[df1.Waiting_Time > lower1]
#df1 = df1[df1.Waiting_Time < upper1]


###############  OUTLIERS EMERGENCY LEVEL 2
mean2 = df2.groupby("Emergency_Stage", as_index=False).mean()['Waiting_Time'][0]
std2 = df2.groupby("Emergency_Stage",as_index=False).std()['Waiting_Time'][0]

cut_off2 = std2*3
lower2, upper2 = 0, mean2 + cut_off2 #mean-cut_off would be <0

# identify outliers df2
outliers2 = [x for x in df2['Waiting_Time'] if x < lower2 or x > upper2]



###############  OUTLIERS EMERGENCY LEVEL 3
mean3 = df3.groupby("Emergency_Stage", as_index=False).mean()['Waiting_Time'][0]
std3 = df3.groupby("Emergency_Stage",as_index=False).std()['Waiting_Time'][0]

cut_off3 = std3*3
lower3, upper3 = 0, mean3 + cut_off3 #mean-cut_off would be <0

# identify outliers df3
outliers3 = [x for x in df3['Waiting_Time'] if x < lower3 or x > upper3]



###############  OUTLIERS EMERGENCY LEVEL 4
mean4 = df4.groupby("Emergency_Stage", as_index=False).mean()['Waiting_Time'][0]
std4= df4.groupby("Emergency_Stage",as_index=False).std()['Waiting_Time'][0]

cut_off4 = std4*3
lower4, upper4 = 0, mean4 + cut_off4 #mean-cut_off would be <0

# identify outliers df4
outliers4 = [x for x in df4['Waiting_Time'] if x < lower4 or x > upper4]



df1 = df1.groupby(["Acquisition_Time","Emergency_Stage","Service","Urgency_Type"], as_index=False).mean()
df2 = df2.groupby(["Acquisition_Time","Emergency_Stage","Service","Urgency_Type"], as_index=False).mean()
df3 = df3.groupby(["Acquisition_Time","Emergency_Stage","Service","Urgency_Type"], as_index=False).mean()
df4 = df4.groupby(["Acquisition_Time","Emergency_Stage","Service","Urgency_Type"], as_index=False).mean()

df1 = df1.drop(columns=["Hospital"])
df2 = df2.drop(columns=["Hospital"])
df3 = df3.drop(columns=["Hospital"])
df4 = df4.drop(columns=["Hospital"])

df1 = df1.set_index('Acquisition_Time')
df2 = df2.set_index('Acquisition_Time')
df3 = df3.set_index('Acquisition_Time')
df4 = df4.set_index('Acquisition_Time')


idx = pd.date_range('11-15-2017', '04-29-2019', freq='D') #freq-'H'


df1_c = df1.query("Service == 'Cirurgia'")
df1_m = df1.query("Service == 'Medicina'")

df1_c= df1_c.reindex(idx)
df1_m= df1_m.reindex(idx)

#df1['Hour'] = df1.index.hour
df1_c['Weekday'] = df1_c.index.weekday
df1_c['Day'] = df1_c.index.day
df1_c['Month'] = df1_c.index.month

df1_c['date_offset'] = (df1_c.index.month*100 + df1_c.index.day - 320)%1300
df1_c['Season'] = pd.cut(df1_c['date_offset'], [0, 300, 602, 900, 1300], 
                      labels=['spring', 'summer', 'autumn', 'winter'])
df1_c=df1_c.drop(columns=['date_offset'])

df1_c = df1_c.fillna(method='ffill')

df1_m['Weekday'] = df1_m.index.weekday
df1_m['Day'] = df1_m.index.day
df1_m['Month'] = df1_m.index.month

df1_m['date_offset'] = (df1_m.index.month*100 + df1_m.index.day - 320)%1300
df1_m['Season'] = pd.cut(df1_m['date_offset'], [0, 300, 602, 900, 1300], 
                      labels=['spring', 'summer', 'autumn', 'winter'])
df1_m=df1_m.drop(columns=['date_offset'])

df1_m = df1_m.fillna(method='ffill')

df1 = pd.concat([df1_c, df1_m])
df1 = df1.sort_index(ascending=True)

df1.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns1.csv', index=False)
df1.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns1_index.csv', index=True)

df1['Acquisition_Time']=df1.index
df1=df1.groupby(['Acquisition_Time','Season'],as_index=False).mean()
df1=df1[df1['Day'].notna()]
df1.index = df1['Acquisition_Time']
df1=df1.drop(columns=['Acquisition_Time'])
df1.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns1_simple.csv', index=True)
df1 = aux.add_yesterday_time(df1)
df1.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns1_simple2.csv', index=True)


df2_c = df2.query("Service == 'Cirurgia'")
df2_m = df2.query("Service == 'Medicina'")

df2_c= df2_c.reindex(idx)
df2_m= df2_m.reindex(idx)

#df2['Hour'] = df2.index.hour
df2_c['Weekday'] = df2_c.index.weekday
df2_c['Day'] = df2_c.index.day
df2_c['Month'] = df2_c.index.month
df2_c['date_offset'] = (df2_c.index.month*100 + df2_c.index.day - 320)%1300
df2_c['Season'] = pd.cut(df2_c['date_offset'], [0, 300, 602, 900, 1300], 
                      labels=['spring', 'summer', 'autumn', 'winter'])
df2_c=df2_c.drop(columns=['date_offset'])
df2_c = df2_c.fillna(method='ffill')

df2_m['Weekday'] = df2_m.index.weekday
df2_m['Day'] = df2_m.index.day
df2_m['Month'] = df2_m.index.month
df2_m['date_offset'] = (df2_m.index.month*100 + df2_m.index.day - 320)%1300
df2_m['Season'] = pd.cut(df2_m['date_offset'], [0, 300, 602, 900, 1300], 
                      labels=['spring', 'summer', 'autumn', 'winter'])
df2_m=df2_m.drop(columns=['date_offset'])
df2_m = df2_m.fillna(method='ffill')

df2 = pd.concat([df2_c, df2_m])
df2 = df2.sort_index(ascending=True)

df2.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns2.csv', index=False)
df2.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns2_index.csv', index=True)

df2['Acquisition_Time']=df2.index
df2=df2.groupby(['Acquisition_Time','Season'],as_index=False).mean()
df2=df2[df2['Day'].notna()]
df2.index = df2['Acquisition_Time']
df2=df2.drop(columns=['Acquisition_Time'])
df2.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns2_simple.csv', index=True)

df2 = aux.add_yesterday_time(df2)
df2.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns2_simple2.csv', index=True)



df3_c = df3.query("Service == 'Cirurgia'")
df3_m = df3.query("Service == 'Medicina'")

df3_c= df3_c.reindex(idx)
df3_m= df3_m.reindex(idx)

#df3['Hour'] = df3.index.hour
df3_c['Weekday'] = df3_c.index.weekday
df3_c['Day'] = df3_c.index.day
df3_c['Month'] = df3_c.index.month
df3_c['date_offset'] = (df3_c.index.month*100 + df3_c.index.day - 320)%1300
df3_c['Season'] = pd.cut(df3_c['date_offset'], [0, 300, 602, 900, 1300], 
                      labels=['spring', 'summer', 'autumn', 'winter'])
df3_c=df3_c.drop(columns=['date_offset'])
df3_c = df3_c.fillna(method='ffill')

df3_m['Weekday'] = df3_m.index.weekday
df3_m['Day'] = df3_m.index.day
df3_m['Month'] = df3_m.index.month
df3_m['date_offset'] = (df3_m.index.month*100 + df3_m.index.day - 320)%1300
df3_m['Season'] = pd.cut(df3_m['date_offset'], [0, 300, 602, 900, 1300], 
                      labels=['spring', 'summer', 'autumn', 'winter'])
df3_m=df3_m.drop(columns=['date_offset'])
df3_m = df3_m.fillna(method='ffill')

df3 = pd.concat([df3_c, df3_m])
df3 = df3.sort_index(ascending=True)

df3.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns3.csv', index=False)
df3.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns3_index.csv', index=True)

df3['Acquisition_Time']=df3.index
df3=df3.groupby(['Acquisition_Time','Season'],as_index=False).mean()
df3=df3[df3['Day'].notna()]
df3.index = df3['Acquisition_Time']
df3=df3.drop(columns=['Acquisition_Time'])
df3.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns3_simple.csv', index=True)

df3 = aux.add_yesterday_time(df3)
df3.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns3_simple2.csv', index=True)


df4_c = df4.query("Service == 'Cirurgia'")
df4_m = df4.query("Service == 'Medicina'")

df4_c= df4_c.reindex(idx)
df4_m= df4_m.reindex(idx)

#df4['Hour'] = df4.index.hour
df4_c['Weekday'] = df4_c.index.weekday
df4_c['Day'] = df4_c.index.day
df4_c['Month'] = df4_c.index.month
df4_c['date_offset'] = (df4_c.index.month*100 + df4_c.index.day - 320)%1300
df4_c['Season'] = pd.cut(df4_c['date_offset'], [0, 300, 602, 900, 1300], 
                      labels=['spring', 'summer', 'autumn', 'winter'])
df4_c=df4_c.drop(columns=['date_offset'])
df4_c = df4_c.fillna(method='ffill')

df4_m['Weekday'] = df4_m.index.weekday
df4_m['Day'] = df4_m.index.day
df4_m['Month'] = df4_m.index.month
df4_m['date_offset'] = (df4_m.index.month*100 + df4_m.index.day - 320)%1300
df4_m['Season'] = pd.cut(df4_m['date_offset'], [0, 300, 602, 900, 1300], 
                      labels=['spring', 'summer', 'autumn', 'winter'])
df4_m=df4_m.drop(columns=['date_offset'])
df4_m = df4_m.fillna(method='ffill')

df4 = pd.concat([df4_c, df4_m])
df4 = df4.sort_index(ascending=True)

df4.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns4.csv', index=False)
df4.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns4_index.csv', index=True)

df4['Acquisition_Time']=df4.index
df4=df4.groupby(['Acquisition_Time','Season'],as_index=False).mean()
df4=df4[df4['Day'].notna()]
df4.index = df4['Acquisition_Time']
df4=df4.drop(columns=['Acquisition_Time'])
df4.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns4_simple.csv', index=True)

df4 = aux.add_yesterday_time(df4)
df4.to_csv(r'C:\Users\loure\Desktop\Tese\sns dataset\sns4_simple2.csv', index=True)



