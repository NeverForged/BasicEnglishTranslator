# -*- coding: utf-8 -*-
import re
import nltk
import string
import logging
import requests
import numpy as np
import cPickle as pickle
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec
from collections import defaultdict
from nltk import pos_tag, word_tokenize
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import CountVectorizer
headers = {"User-Agent": "Mozilla/5.0"}


def main():
    '''
    Scrapes Basic English books from the internet.
    '''
    try:
        corpus = pickle.load(open('../data/basic_english_corpus.pickle', 'rb'))
    except:
        # books available on basic-english.org
        link = 'http://ogden.basic-english.org/books/'
        lst = [link + 'lilliput.html',
               link + 'keawe.html',
               link + 'andersen.html',
               link + 'andersen2.html',
               link + 'ndersen3.html',
               link + 'andersen4.html',
               link + 'ltmatchgirl.html',
               link + 'tpw1.html',
               link + 'arms1.html',
               link + 'arms2.html',
               link + 'arms3.html',
               link + 'astronomy1.html',
               link + 'astronomy1.html',
               link + 'astronomy2.html',
               link + 'astronomy3.html',
               'http://ogden.basic-english.org/carlanna.html',
               link + 'death.html',
               link + 'death3.html',
               link + 'death5.html',
               link + 'death7.html',
               link + 'goldinsect0.html',
               link + 'meno.html',
               'http://ogden.basic-english.org/oos.html',
               link + 'the3signs.html',
               link + 'mathlogic.html',
               link + 'mathlogic7.html',
               link + 'brreason.html',
               link + 'brreason2.html',
               link + 'brreason3.html',
               link + 'statesug.html',
               link + 'confucius.html',
               link + 'buddha.html',
               link + 'bom0.html',
               link + 'koran100.html']
        # history...
        for i in xrange(1, 13):
            lst.append('http://ogden.basic-english.org/ghos' + str(i) +
                       '.html')
        # bible...
        r = requests.get('http://ogden.basic-english.org/bbe/bbe.html')
        soup = BeautifulSoup(r.content, 'html.parser')
        lst_b = ['']
        for table in soup.findAll('table'):
            for link in table.findAll('a'):
                try:
                    lst_b.append(link.get('href')[:link.get('href')
                                                  .index('#')])
                except:
                    lst_b.append(link.get('href'))
        # printable filtering to avoid errors
        for item in lst_b:
            lst.append('http://ogden.basic-english.org/bbe/' + str(item))
        prt = set(string.printable)
        corpus = []
        length = float(len(lst))
        books_in = 0
        print length
        for i, item in enumerate(lst):
            r = requests.get(item)
            soup = BeautifulSoup(r.content, 'html.parser')
            tags = soup.find_all()
            txt = [filter(lambda x: x in prt, tag.get_text()) for tag in tags]
            if len(txt) >= 2:
                book = Book(txt)
                book.clean()
                book.remove_stop_words()
                books_in += 1
            # add to the corpus
            corpus.append(book)
            print 'scraping {:.2f}%  Books:{}\r'.format(100.0*i/length,
                                                        books_in),

        with open('../data/basic_english_corpus.pickle', 'wb') as handle:
            pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Make a model...
    sentences = tokenize_books(corpus)
    vec_set = get_vectorize_set(corpus)
    data_features = vectorize_books(vec_set)
    model, bigrams = word_2_vec(sentences)
    model.init_sims(replace=True)  # for memory
    words = {}
    make_simple = ['CC',  # CC: conjunction, coordinating
                   'DT',  # DT: determiner
                   'IN',  # IN: preposition or conjunction, subordinating
                   'JJR',  # JJR: adjective, comparative
                   'JJS',  # JJR: adjective, comparative
                   'MD',  # MD: modal auxiliary
                   'NN',  # NN: noun, common, singular or mass
                   'NNS',  # NNS: noun, common, plural
                   'PDT',  # PDT: pre-determiner
                   'PDT',  # PDT: pre-determiner
                   'PRP$',  # PRP$: pronoun, possessive
                   'RB',  # RB: adverb
                   'RBR',  # RBR: adverb, comparative
                   'RBS',  # RBS: adverb, superlative
                   'RP',  # RP: particle
                   'UH',  # UH: interjection
                   'VB',  # VB: verb, base form
                   'VBD',  # VBD: verb, past tense
                   'VBG',  # VBG: verb, present participle or gerund
                   'VBN',  # VBN: verb, past participle
                   'VBP',  # VBP: verb, present tense, not 3rd person sing
                   'VBZ',  # VBZ: verb, present tense, 3rd person singular
                   'WDT',  # WDT: WH-determiner
                   'WP',  # WP: WH-pronoun
                   'WP$',  # WP$: WH-pronoun, possessive
                   'WRB'  # WRB: Wh-adverb
                   ]
    for word in pos_tag(model.wv.vocab.keys()):
        if word[1] in make_simple:
            clean = clean_word(word[0])
            words[clean] = [clean, clean, word[1]]
    for bigram in pos_tag(bigrams):
        if bigram[1] in make_simple:
            words[clean_word(bigram[0])] = [clean_word(bigram[0]),
                                            clean_word(bigram[0]),
                                            bigram[1]]
    with open('../data/basic_english_book_words.pickle', 'wb') as handle:
            pickle.dump(words, handle, protocol=pickle.HIGHEST_PROTOCOL)


def clean_word(word):
    '''
    Moving a lot of functions here for speed
    '''
    check_clean = word.encode('ascii', 'replace')
    return check_clean.strip(string.punctuation).lower()


def tokenize_books(corpus):
    '''
    Tokenize and turn into sentences.
    '''
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []
    for book in corpus:
        sentences += book_to_sentences(book, tokenizer)
    wordy = np.mean([len(sentence) for sentence in sentences])
    sentences = sentences
    return sentences


def get_vectorize_set(corpus):
    '''
    '''
    to_vectorize = []
    # get all the books
    for book in corpus:
        # prepare to vectorize
        [to_vectorize.append(word) for word in book.meaningful_words]
    return to_vectorize


def word_2_vec(sentences):
    '''
    Word to vec operation...
    '''
    adj = ['JJR', 'JJS', 'RB', 'RBR', 'RBS', 'JJ', 'DT', 'CD']
    noun = ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 5     # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)

    print "Training model..."
    model = word2vec.Word2Vec(sentences,
                              workers=num_workers,
                              size=num_features,
                              min_count=min_word_count,
                              window=context,
                              sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # bigrams is a misnomer... it's more an adjective-gram
    phrases = Phrases(sentences)
    make_bigrams = Phraser(phrases)
    bigrams = defaultdict(list)
    fails = 0
    for sentence in make_bigrams[sentences]:
        for item in sentence:
            if '_' in item:
                lst = item.split('_')
                if len(lst) == 2:
                    try:
                        wrds = pos_tag(lst)
                        if wrds[0][1] in adj and wrds[1][1] in noun:
                            nun = wrds[1][0].lower().strip(string.punctuation)
                            adj = wrds[0][0].lower().strip(string.punctuation)
                            bigrams[nun].append(adj)
                    except:
                        pass
    # save here to avoid having a model and no 'bigrams'
    model.save('../model/basic_english_only.bin')
    return model, bigrams


def vectorize_books(vectorize_set):
    '''
    '''
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    data_features = vectorizer.fit_transform(vectorize_set)
    return data_features


def book_to_sentences(book, remove_stopwords=False):
    '''
    Function to split a review into parsed sentences. Returns a
    list of sentences, where each sentence is a list of words
    '''
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(book.text.encode("ascii", "ignore")
                                       .strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(book_to_wordlist(raw_sentence,
                                              remove_stopwords))
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def book_to_wordlist(book_text, remove_stopwords=False):
    '''
    Function to convert a document to a sequence of words,
    optionally removing stop words.  Returns a list of words.
    '''
    words = book_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    return words


class Book:
    '''
    Class handler for books scraped from the net.
    '''
    def __init__(self, book):
        self.title = ''
        self.author = ''
        n = len(book)
        lim = 30
        if lim > n:
            lim = n
        for i, line in enumerate(book[:2]):
            if i == 0:
                self.title = line.strip()
                # print self.title
            if i == 1:
                self.author = line.strip()

        # start set to strip off various lengths of front-matter
        temp_text = "".join(book[2:]).encode('ascii', 'replace')
        # -19350 to strip off the Project Gutenberg donation requests
        self.text = temp_text
        self.clean_text = ''
        self.remove_stop_words()

    def clean(self):
        # Get rid of line breaks.
        x = self.text.split("\n")
        x = [i.strip() for i in x]
        self.clean_text = " ".join(x)

        # Lower case.
        self.clean_text = self.clean_text.lower()

        # Remove punctuations
        prt = set(string.printable)
        self.clean_text = filter(lambda x: x in prt, self.clean_text)
        self.clean_text = self.clean_text.strip(string.punctuation)
        # Remove spaces.
        self.clean_text = self.clean_text.replace("  ", " ")
        while "  " in self.clean_text:
            self.clean_text = self.clean_text.replace("  ", " ")

    def remove_stop_words(self):
        stops = set(stopwords.words("english"))
        words = self.clean_text.split()
        self.meaningful_words = [w for w in words if w not in stops]

    def __getitem__(self, index):
        if index == 0:
            return self.clean_text
        if index == 1:
            return self.title
        if index == 2:
            return self.author

if __name__ == "__main__":
    main()
