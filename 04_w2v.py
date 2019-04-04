# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 02:27:03 2018

@author: Faris Salbari
"""

import time
from datetime import timedelta
import multiprocessing
import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import KeyedVectors

start_time = time.time()

csv = "data/ready/data_train.csv"
df = pd.read_csv(csv, header=0, index_col=None)
df['text'] = df.text.astype(str)

tweets = df.text

def tagging_tweets(tweets,tag):
    result = []
    prefix = tag
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result

all_x_w2v = tagging_tweets(tweets, 'tweet')

cores = multiprocessing.cpu_count()
model_sg = Word2Vec(sg=1, size=100, workers=cores, min_count=3)
model_sg.build_vocab([x.words for x in all_x_w2v])
model_sg.train([x.words for x in all_x_w2v], total_examples=len(all_x_w2v), epochs=20)
    
model_sg.save('model/w2v/model_sg.w2v')

finish_time = time.time()
print("\nFinished. Elapsed time: {}".format(timedelta(seconds=finish_time-start_time)))