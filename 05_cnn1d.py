# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 02:14:51 2018

@author: Faris Salbari
"""

import time
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD, Adagrad, Nadam, RMSprop
from keras import backend as K

start_time = time.time()

csv = "data/ready/data_train.csv"
df = pd.read_csv(csv, header=0, index_col=None)
df['text'] = df.text.astype(str)

x = df.text
y = df.target
SEED = 42

length = []
for w in x:
    length.append(len(w.split()))
print(max(length)) #41

maxlen = 45

model_sg = KeyedVectors.load('model/w2v/model_sg.w2v')
vocab_size = len(model_sg.wv.vocab)
print(vocab_size) #739

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x)
sequences_train = tokenizer.texts_to_sequences(x)
x_seq = pad_sequences(sequences_train, maxlen=maxlen)

embedding_matrix = np.zeros((vocab_size, 100))
for i in range(vocab_size):
    embedding_vector = model_sg.wv[model_sg.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

def get_model_alexnet():
    model = Sequential()
    em = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=maxlen, weights=[embedding_matrix], trainable=False)
    model.add(em)
    
    model.add(Conv1D(96, kernel_size=11, strides=4))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(BatchNormalization())
    
    model.add(Conv1D(256, kernel_size=5, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv1D(384, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(384, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(256, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['accuracy'])
    model.summary()
    
    return model

def get_model_zfnet():
    model = Sequential()
    em = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=maxlen, weights=[embedding_matrix], trainable=False)
    model.add(em)
    
    model.add(Conv1D(96, kernel_size=7, strides=2))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(BatchNormalization())
    
    model.add(Conv1D(256, kernel_size=5, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(1024, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['accuracy'])
    model.summary()
    
    return model

def get_model_vgg16():
    model = Sequential()
    em = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=maxlen, weights=[embedding_matrix], trainable=False)
    model.add(em)
    
    model.add(Conv1D(64, kernel_size=3, strides=1))
    model.add(Activation('relu'))
    model.add(Conv1D(64, kernel_size=3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    
    model.add(Conv1D(128, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Conv1D(256, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(256, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
        
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['accuracy'])
    model.summary()
    
    return model

def get_model_vgg19():
    model = Sequential()
    em = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=maxlen, weights=[embedding_matrix], trainable=False)
    model.add(em)
    
    model.add(Conv1D(64, kernel_size=3, strides=1))
    model.add(Activation('relu'))
    model.add(Conv1D(64, kernel_size=3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    
    model.add(Conv1D(128, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Conv1D(256, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(256, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['accuracy'])
    model.summary()
    
    return model

def get_model_lenet():
    model = Sequential()
    em = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=maxlen, weights=[embedding_matrix], trainable=False)
    model.add(em)
    
    model.add(Conv1D(20, kernel_size=5, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    
    model.add(Conv1D(50, kernel_size=5, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['accuracy'])
    model.summary()
    
    return model

def get_model_3layer():
    model = Sequential()
    em = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=maxlen, weights=[embedding_matrix], trainable=False)
    model.add(em)
    
    model.add(Conv1D(16, kernel_size=3, strides=2))
    model.add(Activation('relu'))
    model.add(Conv1D(16, kernel_size=3, strides=2))
    model.add(Activation('relu'))
    model.add(Conv1D(16, kernel_size=3, strides=2))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['accuracy'])
    model.summary()
    
    return model

adam = Adam(lr=0.0001)
sgd = SGD(lr=0.0001)
adagrad = Adagrad(lr=0.0001)
nadam = Nadam(lr=0.0001)
rmsprop = RMSprop(lr=0.0001)

model = get_model_3layer()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
i=0
for train_index, val_index in skf.split(x_seq, y):
    x_train, x_val = x_seq[train_index], x_seq[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    print("\nFold ", i+1)
    history = model.fit(x_train, y_train, batch_size=68, validation_data=(x_val, y_val), epochs=10)
    
    i += 1

model.save('model/cnn/3layer1_nadam00001.h5')

finish_time = time.time()
print("\nFinished. Elapsed time: {}".format(timedelta(seconds=finish_time-start_time)))

del model
K.clear_session()