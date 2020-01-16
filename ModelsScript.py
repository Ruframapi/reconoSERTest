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


import findspark
findspark.init()


import pyspark as ps
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, FloatType, IntegerType, ArrayType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import CountVectorizer, Tokenizer, HashingTF, StandardScaler, Normalizer
from pyspark.ml.feature import StopWordsRemover


#Variable de session de spark
spark = ps.sql.SparkSession.builder.master("local[4]").appName("testreconoser").config("spark.executor.heartbeatInterval","60s").config("spark.network.timeout","61s").getOrCreate()
sc = spark.sparkContext


df = spark.read.option("multiline","true").json("UserFilter50-Training-Random3.json")
df.printSchema()


# Se extraen las etiquetas POS
def pos_etiquetas(s):
	return [i[1] for i in nltk.pos_tag(s)]

pos_tagger_udf = udf(pos_etiquetas, ArrayType(StringType()))
df_pos = df.withColumn('POS', pos_tagger_udf(df['reviews']))


# Se generan los skipgrams
def skip_grams(s):
	grams = []
	for i in skipgrams(s, 2, 2):
		grams.append(str(i))
	return grams

skip_grams_udf = udf(skip_grams, ArrayType(StringType()))
df_skip = df_pos.withColumn('skip_grams', skip_grams_udf(df_pos['POS']))

# Se filtran y se mantienen los ngramas mas comunes
def skip_grams_filter(s):
	return [i for i in s if i in ngrams]

filter_skips_udf = udf(skip_grams_filter, ArrayType(StringType()))
df_skip = df_skip.withColumn('com_skips', filter_skips_udf(df_skip['skip_grams']))

# Se filtran las stop words
def stop_words_filter(s):
	return [i for i in s if i in stops]

stop_words_udf = udf(stop_words_filter, ArrayType(StringType()))
df_stop = df_skip.withColumn('stop_words', stop_words_udf(df_skip['reviews']))

#Feature Vector ngrams+word
def concat(type):
	def concat_(*args):
		return list(chain.from_iterable((arg if arg else [] for arg in args)))
	return udf(concat_, ArrayType(type))

concat_arrays_udf = concat(StringType())
df_feature = df_stop.select("user", concat_arrays_udf("stop_words", "com_skips"))


# Count Vectorize function over the feature vector
hashingTF = HashingTF(numFeatures=285, inputCol='concat_(stop_words, com_skips)', outputCol='features')
tf1 = hashingTF.transform(df_feature)


# Normalize the counts so that they are a percentage of total counts of the features
tf_norm1 = Normalizer(inputCol="features", outputCol="features_norm", p=1).transform(tf1)

# Standardize the vector based on average use of each feature among all users
stdscaler = StandardScaler(inputCol='features_norm', outputCol='scaled', withMean=True)
scale_fit1 = stdscaler.fit(tf_norm1)
scaled1 = scale_fit1.transform(tf_norm1)

