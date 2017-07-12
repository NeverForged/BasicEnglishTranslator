import logging
import sys
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk.data
from gensim.models import word2vec
import gensim as gensim
import cPickle as pickle


def main(num_features=300,
         min_word_count=5,
         num_workers=4,
         context=10,
         downsampling = 1e-3):
    '''
    Function to turn my sentences into a model.
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    '''
    sentences = []
    sentences = pickle.load(open('../data/sentences.pickle', 'rb'))

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

    # Initialize and train the model (this will take some time)

    model = word2vec.Word2Vec(sentences, workers=num_workers,
                              size=num_features, min_count = min_word_count,
                              window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    # model.init_sims(replace=True)
    # since I will remake rather than retrain...

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = (str(num_features) + 'features_' +
                  str(min_word_count) + 'min_word_count_' +
                  str(context) + 'context.npy')

    model.save('../model/'+model_name)
    # model = gensim.models.KeyedVectors.load_word2vec_format(
    #         '../model/300features_4min_word_count_10context.npy', binary=True)
    return model


if __name__ == '__main__':
    # num_features = 300    # Word vector dimensionality
    # min_word_count = 40   # Minimum word count
    # num_workers = 4       # Number of threads to run in parallel
    # context = 10          # Context window size
    # downsampling = 1e-3   # Downsample setting for frequent words
    a = len(sys.argv)
    if a <= 1:
        main()
    if a == 2:
        main(num_features=int(sys.argv[1]))
    elif a == 3:
        main(num_features=int(sys.argv[1]),
             min_word_count=int(sys.argv[2]))
    elif a == 4:
        main(num_features=int(sys.argv[1]),
             min_word_count=int(sys.argv[2]),
             num_workers=int(sys.argv[3]))
    elif a == 5:
        main(num_features=int(sys.argv[1]),
             min_word_count=int(sys.argv[2]),
             num_workers=int(sys.argv[3]),
             context=int(sys.argv[4]))
    elif a == 6:
        main(num_features=int(sys.argv[1]),
             min_word_count=int(sys.argv[2]),
             num_workers=int(sys.argv[3]),
             context=int(sys.argv[4]),
             downsampling=int(sys.argv[5]))
