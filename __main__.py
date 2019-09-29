import pandas as pd
import numpy as np
import nltk
import math
import json, ast
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


def wordVectorize(df_local):
    #for creating feature vector and preprocessing of data
    corpus = df_local['Message'].values.tolist()
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)
    feature_lst = cv.get_feature_names()

    return X, feature_lst

def plotSparsity(df_X_Mat_local):
    #plot the sparsity of the pandas dataframe
    plt.spy(df_X_Mat_local)
    plt.title("Sparse Matrix")
    plt.show()


def TF_IDF(df_X_Mat_local):
    #calculate the tf and idf matrix for the given feature matrix
    df_freq = df_X_Mat.sum(axis = 1, skipna = True)
    df_docs = df_X_Mat.sum(axis = 0, skipna = True)
    
    new_df_freq = df_X_Mat_local.div(df_freq, axis = 0)
    new_df_docs = np.log10(df_X_Mat_local.shape[0]/df_docs)

    new_tf_idf = new_df_freq.mul(new_df_docs, axis = 1)
    
    return new_tf_idf

def train_test_split(df_X_final, df_Y_final):

    print(df_X_final)
    print(df_Y_final)




df = pd.read_csv('spam.csv', '\t')

df.loc[df['Type'] == 'ham', 'Type'] = 0
df.loc[df['Type'] == 'spam', 'Type'] = 1

X_Mat, feature_Vec_lst = wordVectorize(df)

df_X_Mat = pd.DataFrame(X_Mat.todense())

#plotSparsity(df_X_Mat)

#The final value of the feature vector for N emails
#constitute a matrix that would be used further for 
#training the logistic regressio model

df_X_final = TF_IDF(df_X_Mat)

train_test_split(df_X_final, df['Type'])
