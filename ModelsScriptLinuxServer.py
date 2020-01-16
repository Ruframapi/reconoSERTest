import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import nltk
import csv
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.util import skipgrams
from scipy.cluster import hierarchy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity



with open('commonngrams.csv', 'r') as f:
    reader = csv.reader(f)
    ngrams_skips = list(reader)

ngrams = ngrams_skips[0]

stops = stopwords.words('english')
x = [i.split("'") for i in stops]
stops = list(set([i[0] for i in x]))
slang_stops = ['gonna', 'coulda', 'shoulda','lotta', 'lots', 'oughta', 'gotta', 'ain', 'sorta', 'kinda', 'yeah', 'whatever', 'cuz', 'ya', 'haha', 'lol', 'eh']
puncts = ['!', ':', '...', '.', '%', '$', "'", '"', ';']
formattings = ['##', '__', '_', '    ', '*', '**','/']
stops.extend(slang_stops)
stops.extend(puncts)
stops.extend(formattings)



#Funciones de procesado
def pos_etiquetas(s):
    return [i[1] for i in nltk.pos_tag(s['reviews'])]

def skip_grams(s):
    grams = []
    for i in skipgrams(s['pos'], 2, 2):
        grams.append(str(i))
    return grams

def skip_grams_filter(s):
    return [i for i in s["skip_grams"] if i in ngrams]

def stop_words_filter(s):
    return [i for i in s["reviews"] if i in stops]

def concat(s):
    return s['stop_words'] + s['com_skips']

def dummy(doc):
    return doc

def reduce_reviews(s,mcs):
    total = []
    for review in s["reviews"]:
        total.extend(review)
        #print(len(total))
        if len(total) >= mcs:
            return total
    return total

filter_size = [10, 30, 50]
min_comment_size_total = [10, 50, 100, 300, 500, 1000, 200000]


for filter in filter_size:
    filename_training = "UserFilter"+str(filter)+"-Training"
    filename_testing = "UserFilter"+str(filter)+"-Testing"
    df = pd.read_json(filename_training+".json")
    
    ##Aplicacion de funciones
    df['pos'] = df.apply(pos_etiquetas, axis=1)
    df['skip_grams'] =  df.apply(skip_grams, axis=1)
    df['com_skips'] =  df.apply(skip_grams_filter, axis=1)
    df['stop_words'] =  df.apply(stop_words_filter, axis=1)
    df['feature'] = df.apply(concat, axis=1)

    ##Vectorizacion
    cv = CountVectorizer(tokenizer=dummy,preprocessor=dummy)  
    X = cv.fit_transform(df["feature"].to_numpy())
    matrix_cv = normalize(X.toarray())
    scaler = StandardScaler()
    scaler.fit(matrix_cv)
    df["feature_vector"] = list(scaler.transform(matrix_cv))

    #Save training file with vectors
    df.drop(['reviews', 'pos', 'skip_grams', 'com_skips', 'stop_words'], axis=1, inplace=True)
    df.to_pickle(filename_training+"-processing")
    #df = pd.read_pickle(filename_training+"-processing")
    
    for min_comment_size in min_comment_size_total:
    
        print("-----Testing-----")
        df = pd.read_json(filename_testing+".json")
        df["reviews"] = df.apply(reduce_reviews, axis=1, mcs=min_comment_size)
        df.head()

        ##Aplicacion de funciones
        df['pos'] = df.apply(pos_etiquetas, axis=1)
        df['skip_grams'] =  df.apply(skip_grams, axis=1)
        df['com_skips'] =  df.apply(skip_grams_filter, axis=1)
        df['stop_words'] =  df.apply(stop_words_filter, axis=1)
        df['feature'] = df.apply(concat, axis=1)

        X = cv.transform(df["feature"].to_numpy())
        matrix_cv = normalize(X.toarray())
        scaler.transform(matrix_cv)
        df["feature_vector"] = list(scaler.transform(matrix_cv))

        #Save testing file with vectors
        df.drop(['reviews', 'pos', 'skip_grams', 'com_skips', 'stop_words'], axis=1, inplace=True)
        df.to_pickle(filename_testing+"-processing"+str(min_comment_size))

        #Load pickle testing and traning
        print("--Loading Files--")
        df = pd.read_pickle(filename_training+"-processing")
        df_test = pd.read_pickle(filename_testing+"-processing"+str(min_comment_size))

        #Cosine similarities
        similarities = cosine_similarity(df_test["feature_vector"].to_list(),df["feature_vector"].to_list())
        pdf = pd.DataFrame(similarities)
        pdf.to_pickle(filename_testing+"-similarities"+str(min_comment_size))