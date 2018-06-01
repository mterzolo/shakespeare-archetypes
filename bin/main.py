import gensim
import string
import logging
import os
import pickle
import pandas as pd
from random import shuffle
from sklearn.cluster import KMeans

import lib


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.INFO)

    documents, valid_chars = extract()
    w2v_model, embedding_matrix = transform(documents, valid_chars)
    archetypes = cluster(embedding_matrix, valid_chars)
    load(documents, embedding_matrix, w2v_model, archetypes)
    pass


def extract():
    """
    Extract necessary data / resources from upstream. This method will:
     - Read in data from plays and sonnets
     - cursory cleaning and concatenation
     - Validate which characters we will group together
    :return: documents, valid_chars
    :rtype: (list, list)
    """

    logging.info('Begin extract')

    # Read in csv containing data on plays
    plays = pd.read_csv('..//data//Shakespeare_data.csv')
    play_lines = list(plays['PlayerLine'].astype(str))

    logging.info('Original number of lines in play data: {}'.format(len(play_lines)))

    # Collect characters we will cluster
    valid_chars = plays.groupby(['Player']).count()
    valid_chars = valid_chars['Play'].sort_values(ascending=False)
    valid_chars = list(valid_chars[0:70].index)
    valid_chars = [i.lower() for i in valid_chars if len(i.split()) == 1]

    logging.debug('Number of valid charachters: {}'.format(len(valid_chars)))

    # Read in txt file containing sonnets
    text_file = open("..//data//sonnets.txt", "r")
    sonnet_lines = text_file.readlines()

    logging.debug('Original number of lines in sonnet data: {}'.format(len(sonnet_lines)))

    # Cursory cleaning of sonnet data
    sonnet_lines = [i for i in sonnet_lines if len(i) > 10]
    sonnet_lines = [i.lstrip() for i in sonnet_lines]
    sonnet_lines = [i.replace('\n', '') for i in sonnet_lines if len(i) > 10]

    logging.debug('Cleaned number of lines in sonnet data: {}'.format(len(sonnet_lines)))

    # Concatenate data sets
    documents = play_lines + sonnet_lines

    logging.debug('Total number of lines for model: {}'.format(len(documents)))

    return documents, valid_chars


def transform(documents, valid_chars):
    """

    Prepare data for word2vec and create the embeddings for the documents
    :param documents: all text from shakespeare
    :param valid_chars: all characters that we can cluster together
    :return: word2vec model, embedding matrix
    """
    logging.info('Begin transform')

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    documents = [i.translate(translator) for i in documents]

    # Shuffle to break up play and sonnet lines
    shuffle(documents)

    # Remove stopwords
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

    logging.info('Begin word2vec model fitting')

    # Create word embeddings
    w2v_model = gensim.models.Word2Vec(texts, size=300, window=10, min_count=2, workers=2, sg=1)

    # Create embedding matrix for valid characters
    embedding_matrix = w2v_model[valid_chars]

    return w2v_model, embedding_matrix


def cluster(embedding_matrix, valid_chars):
    """

    Clusters characters together based on their embedding vectors
    :param embedding_matrix: w2v embeddings for selected characters
    :param valid_chars: all characters that we can cluster together
    :return: clusters of characters that represent archetypes
    """
    logging.info('Begin clustering')

    # Predict cluster labels
    kmeans_model = KMeans(n_clusters=6).fit(embedding_matrix)
    labels = kmeans_model.predict(embedding_matrix)

    # Save to dataframe
    archetypes = pd.DataFrame({'Name': valid_chars,
                               'cluster': labels}).sort_values('cluster')

    return archetypes


def load(documents, valid_chars, w2v_model, archetypes):

    logging.info('Begin load')

    logging.info('Writing documents to file')
    pickle.dump(documents, open(os.path.join(lib.get_batch_output_folder() + '/documents.pkl'), 'wb'))

    logging.info('Writing valid_chars to file')
    pickle.dump(valid_chars, open(os.path.join(lib.get_batch_output_folder() + '/valid_chars.pkl'), 'wb'))

    logging.info('Writing w2v_model to file')
    pickle.dump(w2v_model, open(os.path.join(lib.get_batch_output_folder() + '/w2v_model.pkl'), 'wb'))

    logging.info('Writing archetypes observations to txt file ')
    archetypes.to_csv(os.path.join(lib.get_batch_output_folder(), 'archetypes.csv'))
    logging.info('End load')
    pass


# Main section
if __name__ == '__main__':
    main()