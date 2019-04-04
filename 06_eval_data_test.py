# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 02:33:04 2018

@author: Faris Salbari
"""

import pandas as pd
from keras.models import load_model
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

csv = "data/ready/data_test.csv"
df = pd.read_csv(csv, header=0, index_col=None, usecols=['text', 'target'])
df['text'] = df.text.astype(str)

x_test = df.text
y_test = df.target
maxlen = 45

model_sg = KeyedVectors.load('model/w2v/model_sg.w2v')
vocab_size = len(model_sg.wv.vocab)

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_test)
sequences_test = tokenizer.texts_to_sequences(x_test)
x_test_seq = pad_sequences(sequences_test, maxlen=maxlen)

cnn = "model/cnn/3layer1_nadam00001.h5"
model = load_model(cnn)

print("\nScore on Test data")
score = model.evaluate(x_test_seq, y_test)
accuracy = str(score[1]*100)
loss = str(score[0])
print("Accuracy: ", accuracy)
print("Total loss: ", loss)

preds = model.predict_classes(x_test_seq)
df = df.assign(prediction=preds) 
df.to_csv("data/result/3layer1_nadam00001_%sacc_%sloss.csv"%(accuracy, loss), sep=';')