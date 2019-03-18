import pandas as pd
import pickle
import numpy as np
import itertools
from helper_functions import read_pickle, dict2list
!pwd
#! /usr/bin/env python3


subs = np.array(['science', 'funny', 'engineering', 'compsci',
                'machinelearning', 'datascience', 'math', 'statistics'])
subs = np.array(['science', 'funny', 'python','machinelearning', 'math', 'statistics'])


df_data.to_csv('reddit_thread_label.csv')


df_data.head()
df_data.to_pickle('reddit_thread_label_stem_train')
df_data.shape
subs = np.array(['science', 'funny'])

seed = 3000
seed = np.random.randint(low=0,high= 27000,size=(3000,))
test_seed=np.array(documents)
documents.shape
test_seed[~seed]train_seed=documents[seed]

data_path = 'data/reddit_r_' + sub + '.pk'
data_dict = read_pickle(data_path)
documents = np.array(dict2list(data_dict))
documents[seed].shape
np.delete(documents, seed).shape

def load_tokensize(subs):
    # Loads multiple subreddits
    # ['science','funny','engineering','machine_learning','math','statistics']
    corpus = []
    bool_mat_all = []


    count = 0
    for sub in subs:
        # Load data from each sub and then put into corpus
        # Generate a label for each document to keep track of classs
        data_path = 'data/reddit_r_' + sub + '.pk'
        data_dict = read_pickle(data_path)
        documents = np.array(dict2list(data_dict))
        #documents = documents[~seed]
        documents = np.delete(documents, seed)
        corpus.append(documents)

        # Make a zero matrix for one hot encoding
        bool_mat = np.zeros(shape=(len(documents),len(subs)))
        bool_mat[:,subs==sub]=1

        try:
            bool_mat_all = np.vstack((bool_mat_all,bool_mat))
        except:
            bool_mat_all=bool_mat

        count += 1
    corpus = list(itertools.chain(*corpus))

    df_data = pd.DataFrame(data=bool_mat_all,dtype=int)
    col_str = subs
    df_data.columns = col_str
    df_data['documents'] = pd.Series(data=np.array(corpus))
    return df_data
