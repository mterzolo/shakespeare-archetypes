def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.INFO)

    documents, valid_chars = extract()
    w2v_model, embedding_matrix = transform(documents, valid_chars)
    embeddings = model(w2v_model, embedding_matrix, valid_chars)
    load(embeddings, documents)
    pass

def extract():
    """
    Extract necessary data / resources from upstream. This method will:
     - Read in data from plays and sonnets
     - cursory cleaning and concatenation
     - Validate which characters we will group together
    :return: documents, valid_chars
    :rtype: (numpy.array, numpy.array)
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
    :return: word2vec model, embedding matrix for selected characters
    """
    logging.info('Begin transform')

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    documents = [i.translate(translator) for i in documents]

    # Remove stopwords
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

    logging.info('Begin word2vec model fitting')

    # Create word embeddings
    w2v_model = gensim.models.Word2Vec(texts, size=300, window=10, min_count=2, workers=2, sg=1)

    # Create embedding matrix for valid characters
    embedding_matrix = w2v_model[valid_chars]

    return w2v_model, embedding_matrix


# Main section
if __name__ == '__main__':
    main()