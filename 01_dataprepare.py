# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 00:54:47 2018

@author: Faris Salbari
"""

import pandas as pd
from pprint import pprint

file_0 = 'data/0.csv'
file_1 = 'data/1.csv'

def load_and_label(file_):
    df = pd.read_csv(file_, header=0, index_col=None, encoding='latin1',sep=';')
    df.drop(["username","date","retweets","favorites","geo","mentions","hashtags","id","permalink"],axis=1,inplace=True)
    if (file_ is file_0):
        df = df.assign(depresi = 0)
    elif (file_ is file_1):
        df = df.assign(depresi = 1)
    return df

df_0 = load_and_label(file_0)
df_1 = load_and_label(file_1)
df = [df_0, df_1]
df_join = pd.concat(df, axis=0, ignore_index=True)
df_join.to_csv('data/all.csv',encoding='utf-8')

df_join = pd.read_csv("data/all.csv", header=0, index_col=None)
df_join['text'] = df_join.text.astype(str)
df_join['pre_clean_len'] = [len(t) for t in df_join.text]

data_dict = {
        'text':{
                'type':df_join.text.dtype,
                'description':'tweet text'
                },
        'depresi':{
                'type':df_join.depresi.dtype,
                'description':'0:negative, 1:positive'
                },
        'pre_clean_len':{
                'type':df_join.pre_clean_len.dtype,
                'description':'Length of the tweet before cleaning'
                },
        'dataset_shape':df_join.shape
        }
        
pprint(data_dict)