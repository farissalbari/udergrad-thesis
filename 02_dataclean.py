# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 00:54:35 2018

@author: Faris Salbari
"""

import time
from datetime import timedelta
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

start_time = time.time()

df = pd.read_csv("data/all.csv", header=0, index_col=None)
df['text'] = df.text.astype(str)
swfactory = StopWordRemoverFactory()
stopword = swfactory.create_stop_word_remover()
stfactory = StemmerFactory()
stemmer = stfactory.create_stemmer()
tok = WordPunctTokenizer()

removehtml = r'http\S+'
removemention = r'@\S+'
removewww = r'www.\S+'
removepic = r'pic\S+'

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    soup1 = BeautifulSoup(souped, 'html.parser')
    souped1 = soup1.get_text()
    stripped = re.sub(removehtml,'',souped1)
    stripped = re.sub(removemention,'',stripped)
    stripped = re.sub(removewww, '', stripped)
    stripped = re.sub(removepic, '', stripped)
    letters_only = re.sub('[^a-zA-Z]',' ',stripped)
    stopw = stopword.remove(letters_only)
    stemmed = stemmer.stem(stopw)
    words = [x for x in tok.tokenize(stemmed) if len(x) > 2]
    return (" ".join(words)).strip()

print("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in range(0,1000):
    if((i+1)%100 == 0):
        print("Tweets %d of %d has been processed" % (i+1, 1000))
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))
    
clean_df = pd.DataFrame(clean_tweet_texts, columns=['text'])
clean_df['target'] = df.depresi
clean_df.to_csv('data/clean_all.csv',encoding='utf-8')

finish_time = time.time()
print("\nFinished. Elapsed time: {}".format(timedelta(seconds=finish_time-start_time)))