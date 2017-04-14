# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:59:23 2017

@author: keyur
"""
# In[]:
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import *
from surprise.accuracy import rmse, mae, fcp
from UserDefine import PredictMean
from sklearn.grid_search import ParameterGrid
import pickle
from os import listdir

# In[]

#read data from file 
def parse(path):
	g = open(path, 'rb')
	for l in g:
		yield eval(l)
        
#parse data into a pandas dataframe
def get_df(path):
	i = 0
	df = {}
	for d in parse(path):
		df[i] = d
		i += 1
	return pd.DataFrame.from_dict(df, orient='index')

# if the data has been parsed:
#   load it 
# else:
#   parse it, save it and load it
def possibly_get_df(path):
    try:
        df = pd.read_csv('use_item_ratings.csv')
    except IOError:
        df = get_df(path)
        df.to_csv('use_item_ratings.csv', index=False)
    return df

df = possibly_get_df('Clothing_Shoes_and_Jewelry_5.json')
df.head()

# In[]

ratings_df = df[['reviewerID', 'asin','overall']]
print "Dataframe after dropping unneccessary fields:"
ratings_df.head()

# In[]

#number of rows in dataframe  which are the total number of ratings
total_num_of_ratings = len(ratings_df)
print "total number of user-items ratings is: %d" % total_num_of_ratings

#number of unique reviewers
num_of_users = len(ratings_df['reviewerID'].unique())
print "number of unique reviewers is: %d" % num_of_users

#number of unique items
num_of_items = len(ratings_df['asin'].unique())
print "number of unique items is: %d" % num_of_items

#average rating and standar deviation
ratings_mean = ratings_df['overall'].mean()
ratings_std = ratings_df['overall'].std()
print "average rating is %.3f and standard deviation %.3f" % (ratings_mean,ratings_std)

# The sparsity of the user-item matrix
sparsity = float(total_num_of_ratings) / (num_of_users * num_of_items)
print "sparisity of the user-item matrix is %.8f" % sparsity

# In[]
users_df = ratings_df.groupby('reviewerID').size()
print "The average number of items rated by a user is %f" % np.mean(users_df)
print "The average number of items rated by a user is %f" % np.std(users_df)
print "The minimum number of items rated by a user is %d" % min(users_df)
print "The maximum number of items rated by a user is %d" % max(users_df)

# In[]
items_df = ratings_df.groupby('asin').size()
print "The average number of users that rated an item is %f" % np.mean(items_df)
print "The standard deviation of number of users that rated an item is %f" % np.std(items_df)
print "The minimum number of users that rated an item is %d" % min(items_df)
print "The minimum number of users that rated an item is %d" % max(items_df)

# In[]
# frequency of each rating in a histogram
plt.figure(1)
plt.hist(ratings_df['overall'],5, facecolor='g')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.axis([0, 5, 0, 180000])

# percentage of each rating in a pie chart
plt.figure(2)
labels = 'rating of 1', 'rating of 2' ,'rating of 3', 'rating of 4','rating of 5'
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','red']
plt.pie(ratings_df.groupby('overall').size(),labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')

plt.show()

# In[]:
ratings_df.to_csv('cleaned_user_item_ratings.csv', index=False,header=False)

# In[]

# path to dataset file
original_file_path = 'cleaned_user_item_ratings.csv'

reader = Reader(line_format='user item rating', sep=',')
original_data = Dataset.load_from_file(original_file_path, reader=reader)

#Split data into 4 folds, 3 for training and 1 for testing.
original_data.split(n_folds=4)

# In[]

# Method that will be used for all algorithm.
# Input: Algorithm to use
#       Dataset
# Output: Error measurements
def algorithm_to_results(data, algo):
    algo = algo()
    alogPerf = evaluate(algo, data,measures=['RMSE', 'FCP'], verbose=False)
    return alogPerf

# In[]

print algorithm_to_results(original_data,PredictMean)

# In[]

print algorithm_to_results(original_data, SVD)

# In[]

print algorithm_to_results(original_data, BaselineOnly)

# In[]

print algorithm_to_results(original_data, CoClustering)

# In[]
