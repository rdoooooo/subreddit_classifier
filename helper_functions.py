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


def read_pickle(data_path):
    # Reads pickle file and puts into dictionary
    data_dict = {}  # Create an empty dictionary
    with open(data_path, 'rb') as f:
        data_dict.update(pickle.load(f))
    return data_dict


def load_tokensize(model, subs=['science', 'funny', 'engineering']):
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
        if (len(doc_token) < 3) or (doc_token == []):
            continue
        tokens_clean.append([token for token in doc_token if len(token) > 2])
    return tokens_clean


def word2vec():
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


def doPCA(sentvec, n_components=5):
    pca = PCA(n_components=n_components)
    pca.fit_transform(sentvec)
    print(f'Explained variance {pca.explained_variance_}')
    return pca


def wordvec2sentvec(model, df_data):
    # Converts word2vec to sentence2vec averaging word2vec vectors
    df_data['sentvec'] = df_data.document.apply(
        lambda x: document_vector(model, x))
    drop_index = df_data[df_data.vector.apply(
        lambda x: check_len(x)) == False].index
    df_final = df_data.drop(drop_index, axis=0)
    return df_final


def plot_scatter(sentvecpca, df_sent):
    plt.figure(figsize=(10, 10))
    plt.scatter(x=sentvecpca[:, 0], y=sentvecpca[:, 1],
                c=df_sent.labels.tolist(), alpha=.1, s=50)
    plt.legend(subs)
    plt.show()


if __name__ == '__main__':
    # data_path = 'data/reddit_r_science_funny'
    # data_dict = read_pickle(data_path)
    # documents = dict2list(data_dict)
    # doc_tokens = doc2token(documents)
    subs = ['science', 'funny', 'engineering']
    model = word2vec()
    df_data = load_tokensize(model=model, subs=subs)
    df_sent = wordvec2sentvec(model=model, df_data=df_data)
    sentvec = np.array(df_sent.sentvec.tolist())
    sentvecpca = doPCA(sentvec, n_components=2)


#model.most_similar(positive=['sugar', 'diabetes'], negative=['fat'])
