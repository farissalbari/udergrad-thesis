# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 02:25:16 2018

@author: Faris Salbari
"""

import pandas as pd
from sklearn.model_selection import train_test_split

csv = "data/clean_all.csv"
df = pd.read_csv(csv, header=0, index_col=None)
df['text'] = df.text.astype(str)

x = df.text
y = df.target
SEED = 42

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=SEED)

df_train = pd.DataFrame({'text':x_train, 'target':y_train})
df_train.to_csv('data/ready/data_train.csv',encoding='utf-8')
df_test = pd.DataFrame({'text':x_test, 'target':y_test})
df_test.to_csv('data/ready/data_test.csv',encoding='utf-8')