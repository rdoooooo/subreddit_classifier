import sklearn
import pickle
import pprint as pp
import itertools
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, stem_text
from helper_functions import read_pickle
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import proj3d


def load_tokensize(model, subs):
    # Loads multiple subreddits
    # ['science','funny','engineering','machine_learning','math','statistics']
    corpus = []
    labels = []
    count = 0
    for sub in subs:
        # Load data from each sub and then put into corpus
        # Generate a label for each document to keep track of classs
        data_path = 'data/reddit_r_' + sub + '.pk'
        data_dict = read_pickle(data_path)
        documents = dict2list(data_dict)
        doc_tokens = doc2tokens(documents, model)
        corpus.append(doc_tokens)
        labels.append(np.ones(shape=(len(doc_tokens), 1)) * count)
        count += 1
    corpus = list(itertools.chain(*corpus))
    labels = list(itertools.chain(*labels))
    df_data = pd.DataFrame(data=[corpus, labels])
    df_data = df_data.T
    df_data.columns = ['document', 'labels']
    df_data.labels = df_data.labels.astype(int)
    return df_data


def dict2list(data_dict):
    # Flatten a dictionary to a single list
    listoflist = [item for key, item in data_dict.items()]
    return list(itertools.chain(*listoflist))


def doc2tokens(documents, model):
    # tokenize documents with gensim's tokenize() function
    tokens = [list(gensim.utils.tokenize(doc, lower=True))
              for doc in documents]
    # Build filters to removed stop words
    CUSTOM_FILTERS = [remove_stopwords]
    doc_tokens = [preprocess_string(" ".join(doc), CUSTOM_FILTERS)
                  for doc in tokens]
    tokens_clean = []

    # Make sure tokens all exist in the corpus
    for doc_token in doc_tokens:
        doc_token = [token for token in doc_token if token in model.vocab]
        # if token is shorter than 3 char than dont add to list
        if (len(doc_token) < 3) or (doc_token == []):
            continue
        tokens_clean.append([token for token in doc_token if len(token) > 2])
    return tokens_clean


def word2vec():
    # load googles trained word2vec model
    google_vec_file = 'data/word2vec/GoogleNews-vectors-negative300.bin'
    # Load it!  This might take a few minutes...
    model = gensim.models.KeyedVectors.load_word2vec_format(
        google_vec_file, binary=True)

    # Build own model
    # def word2vec(doc_tokens, size=10, window=2, min_count=1, sg=1, iter=50):
    # model = gensim.models.Word2Vec(
    #     doc_tokens, size=size, window=window, min_count=min_count, sg=sg, iter=iter)
    return model


def document_vector(model, doc):
    # Calculates sentence vector using average or words vectors
    try:
        sentvec = np.mean(model[doc], axis=0)
        return sentvec
    except:
        return 0
    return np.mean(model[x], axis=0)


def check_len(doc):
    # Used as to drop out datasets that are too short
    try:
        if len(doc) == 10:
            return True
    except:
        return False


def doPCA(sentvec, n_components=3):
    pca = PCA(n_components=n_components)
    pca.fit_transform(sentvec)
    results = pca.fit_transform(sentvec)
    print(f'Explained variance {pca.explained_variance_}')
    return pca, results


def doTSNE(sentvec, n_components=2):
    tsne = TSNE(n_components=n_components)
    tsne.fit_transform(sentvec)
    results = tsne.fit_transform(sentvec)
    # print(f'Explained variance {tsne.explained_variance_}')
    return tsne, results


def wordvec2sentvec(model, df_data):
    # Converts word2vec to sentence2vec averaging word2vec vectors
    df_data['sentvec'] = df_data.document.apply(
        lambda x: document_vector(model, x))
    drop_index = df_data[df_data.sentvec.apply(
        lambda x: check_len(x)) == False].index
    df_final = df_data.drop(drop_index, axis=0)
    return df_final


def plot_scatter(data, labels, subs):
    subs = np.unique(subs)
    fig = plt.figure(figsize=(12, 12))
    for label in np.unique(labels):
        logic = labels == label
        subset = data[logic, :]
        plt.scatter(x=subset[:, 0], y=subset[:, 1],
                    alpha=.1, s=100, marker='.')
    lgnd = plt.legend(subs, fontsize=14)

    # change the marker size manually for both lines
    for i in np.arange(len(subs)):
        lgnd.legendHandles[i]._sizes = [100]
    plt.axis('off')
    plt.show()

    return fig


def plot_scatter3d(sentvecpca, labels, subs, angle1=30, angle2=45):
    fig = plt.figure(figsize=(12, 12))

    ax = fig.gca(projection='3d')
    ax.view_init(angle1, angle2)
    # The fix
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    for label in labels:
        logic = labels == label
        subset = sentvecpca[logic]
        ax.scatter(xs=subset[:, 0], ys=subset[:, 1], zs=subset[:, 2],
                   alpha=.1, s=10, marker='.')
    lgnd = plt.legend(subs, fontsize=14)

    # change the marker size manually for both lines
    for i in np.arange(len(subs)):
        lgnd.legendHandles[i]._sizes = [1000]

    plt.show()


def subset_data(labels, sentvec, sub=0, n_samples=100, subs=0):
    # Unique labels
    unique_labels = np.unique(labels)
    # Allocate space of zeros [n_samples*unique_labels, n_dimension]
    # n_dimension are the vector dimensions
    sentvec_subset = np.zeros(
        shape=(n_samples * len(unique_labels), sentvec.shape[1]))
    label_subset = np.zeros(
        shape=(n_samples * len(unique_labels), ))
    for i, label in enumerate(unique_labels):
        # Grab all data with a specific label
        sentvec_label = sentvec[labels == label, :]
        index = np.arange(len(sentvec_label))
        # From a specific group of labels, randomly sample n_samples from the group
        index_subset = np.random.choice(
            index.reshape(-1,), n_samples, replace=False)

        # Figures where to store the new labels
        start = 0 + (i * n_samples)
        stop = start + n_samples

        # Append samples to the sample set
        sentvec_subset[start:stop, :] = sentvec_label[index_subset, :]
        # print(sentvec_label[index_subset,:].shape)

        # Build a label_subset that has the labels for the sentvec_subset
        label_subset[start:stop] = np.ones(shape=(n_samples,)) * i

    # Build a sub_subset
    if sub == 1:

        sub_subset = []
        for i, label in enumerate(np.unique(labels)):
            logic = labels == label
            sub_subset.append([subs[i]] * sum(logic))
        sub_subset = np.array(list(itertools.chain(*sub_subset)))

        return sentvec_subset, label_subset, sub_subset
    else:
        return sentvec_subset, label_subset


def word2vec_model():
    subs = np.array(['science', 'funny', 'engineering',
                     'machinelearning', 'datascience', 'math', 'statistics'])
    model = word2vec()
    df_data = load_tokensize(model=model, subs=subs)
    df_sent = wordvec2sentvec(model=model, df_data=df_data)
    sentvec = np.array(df_sent.sentvec.tolist())
    labels = np.array(df_sent.labels.tolist())
    df_sent['sub'] = pd.Series(data=labels)
    return sentvec, df_sent, subs


if __name__ == '__main__':
    # data_path = 'data/reddit_r_science_funny'
    # data_dict = read_pickle(data_path)
    # documents = dict2list(data_dict)
    # doc_tokens = doc2token(documents)
    # subs = np.array(['science', 'funny', 'engineering', 'compsci',
    #                  'machinelearning', 'datascience', 'math', 'statistics'])
    subs = np.array(['science', 'funny', 'python', 'statistics','machinelearning','datascience'])

    model = word2vec()
    df_data = load_tokensize(model=model, subs=subs)
    df_sent = wordvec2sentvec(model=model, df_data=df_data)
    sentvec = np.array(df_sent.sentvec.tolist())
    labels = np.array(df_sent.labels.values)
    # make a vector with all the sub names
    subs_vec = []
    for i, label in enumerate(np.unique(labels)):
        logic = labels == label
        subs_vec.append([subs[i]] * sum(logic))
    df_sent['sub'] = np.array(list(itertools.chain(*subs_vec)))


    # df_sent.to_pickle('word2vec_output')
    # labels.shape
    # sentvec.shape
    sentvec_subset, label_subset = subset_data(labels, sentvec, n_samples=1000)
    tsne, sentvec_tsne = doTSNE(sentvec_subset, n_components=2)
    fig = plot_scatter(data=sentvec_tsne, labels=label_subset, subs=subs)
    fig.savefig('TSNE_reddit_500_DS.pdf')


    pca, sentvec_pca = doPCA(sentvec_subset, n_components=2)


    fig = plot_scatter(data=sentvec_pca, labels=label, subs=subs)

#
# temp = pd.DataFrame(data=sentvec_tsne)
# temp.to_csv('Reddit_viz.csv')
#
# tsne, sentvec_tsne = doTSNE(sentvec_subset, n_components=3)
#
# plot_scatter3d(sentvec_subset, labels, subs, angle1=30, angle2=45)
