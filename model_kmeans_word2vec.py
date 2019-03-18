import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from model_word2vec import subset_data, doTSNE, plot_scatter
import matplotlib.pyplot as plt
from importlib import reload
import model_word2vec as wv
reload(wv)
# Load data
df_data = pd.read_pickle('word2vec_output')

train_size = 800
test_size = 200

# Subset data to prototype
labels = df_data.labels
sentvec = np.array(df_data.sentvec.tolist())
subs = df_data['sub'].unique()
sentvec_subset, label_subset, subs_subset = subset_data(
    labels, sentvec, subs=subs, sub=1, n_samples=train_size+test_size)
pca, sentvec_pca = doPCA(sentvec_subset, n_components=2)


'''
Calculate Kmeans thread titles (sentence vectors) and see how well the clusters
are able to match that of the truth

Prototype has to run onl a subset of the sentence vectors because too expensive
once it's on aws we will run the entire dataset.
'''

#  Build Kmeans model
# number of clusters, set to number of unique subreddits included
n_clusters = len(np.unique(label_subset))
train = sentvec_pca[:-test_size,:]
test =  sentvec_pca[-test_size:,:]

from sklearn.cluster import KMeans

label_subset = np.array(label_subset)
for unique in np.unique(label_subset):
    less_train = sentvec_subset[label_subset==unique][:-test_size,:]
    less_test = sentvec_subset[label_subset==unique][-test_size::,:]
    less_test_label = label_subset[label_subset==unique][-test_size::]
    print(less_test_label.shape)
    try:
        train_data = np.vstack((train_data,less_train))
        test_data = np.vstack((test_data,less_test))
        test_data_label = np.hstack((test_data_label,less_test_label))
    except:
        train_data = less_train
        test_data = less_test
        test_data_label = less_test_label

kmeans = KMeans(n_clusters=k).fit(train_data)
predict = kmeans.predict(test_data)
sklearn.metrics.accuracy_score(test_data_label, predict)
