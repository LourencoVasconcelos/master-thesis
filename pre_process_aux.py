# -*- coding: utf-8 -*-
"""
@author: Lourenço Vasconcelos
"""


def add_yesterday_time(n_df):
    n_df['Yesterday_Waiting_Time'] = ""
    n_df = n_df.reset_index()
    n_df = n_df.drop(columns=["Acquisition_Time"])
    for row,col in n_df.iterrows():
        if row == 0:
            continue
        else:
            n_df.at[row,'Yesterday_Waiting_Time'] = n_df.at[row-1,'Waiting_Time']
    new_df = n_df[1:]
    return new_df

