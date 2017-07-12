import os
from nltk import pos_tag, word_tokenize
from Book import Book
import sys
import pandas as pd
import unicodedata
import numpy as np
from nltk.corpus import stopwords
from nltk.help import upenn_tagset
from sklearn.feature_extraction.text import CountVectorizer
import string
import logging
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk.data
from gensim.models import Word2Vec
import gensim as gensim
import cPickle as pickle
from BasicEnglishTranslator import BasicEnglishTranslator
import load_model as lm
from nltk import pos_tag, word_tokenize
import time
from gensim.models.phrases import Phrases, Phraser
from collections import defaultdict
import random
from bs4 import BeautifulSoup
import requests

class Author():
    '''
    This class handles loading an author, or creating one if not already
    in our 'authors' folder.

    PARAMETERS
    author: name of the author, as it appears in my file paths.

    overwrite: Default: False.  Set to 'True' to replace the Author class for
        the author specified.  False means we try to load it from a pickle
        file first.

    ATTRIBUTES
    author: Author's name
    author_fromatted: Name formatted as lowercase with '_'
    corpus: a list of Book objects
    basic_list: List of basic_text with parts of speach
    vectorizer: the sklearn CountVectorizer used to vectorize the books in the
        corpus.

    METHODS
    fit(input_text): Main function, builds the various lists and texts off
        of the input_text (string)
    '''

    def __init__(self, author, model=None, overwrite=False, verbose=False,
                 threshold=10):
        '''
        Initializer for Author class.  Checks if we already have an author
        '''
        self.author = author
        a = author.encode('ascii', 'replace')
        a = a.lower().strip().replace(' ','_')
        self.author_fromatted = a
        self.threshold = threshold
        self.verbose = verbose
        self.book_num  = 0
        self.adj =['JJR','JJS','RB', 'RBR','RBS', 'JJ', 'DT', 'CD']
        self.noun = ['NN', 'NNS', 'VB', 'VBD','VBG','VBN', 'VBP', 'VBZ']
        try: # corpus
            self.corpus = pickle.load(open('../authors/' + a + '_corpus', 'rb'))
        except: # get corpus...
            self.corpus = []
            self.get_books()
            self.tokenize_books()
            self.get_vectorize_set()
            self.load_sentences()
            with open('../authors/' + a + '_corpus', 'wb') as handle:
                    pickle.dump(self.corpus, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            with open('../authors/' + a + '_wordy', 'wb') as handle:
                    pickle.dump(self.wordy, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
        self.words = {} # initialize...
        try:
            self.words = pickle.load(open('../authors/' + a + '_words', 'rb'))
        except:
            print 'Need to make a dictionary...'
            self.tokenize_books()
            self.get_vectorize_set()
            self.vectorize_books()
            try:
                print '   ...checking for a model...'
                self.author_model = gensim.models.KeyedVectors.load_word2vec_format('../model/' + a + '.bin')
                print '      ...model found.'
            except:
                print '      ...none found, building...'
                self.author_model = self.word_2_vec()
                print '         ...done.'
            print '   ....creating the dictionary...'
            for word in pos_tag(self.author_model.wv.vocab.keys()):
                self.words[word[0]] = [word[0], word[0], word[1]]
            with open('../authors/' + a + '_words', 'wb') as handle:
                    pickle.dump(self.words, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            with open('../authors/' + a + '_wordy', 'wb') as handle:
                      pickle.dump(self.wordy, handle,
                                  protocol=pickle.HIGHEST_PROTOCOL)
            self.load_sentences()
            print '   ...done.'
        #b = '../model/GoogleNews-vectors-negative300.bin'
        if model == None:
            print '   ...no model specified, loading...'
            b = '../model/300features_5min_word_count_10context.npy'
            self.model =  gensim.models.KeyedVectors.load_word2vec_format(b)
            print '      ...done.'
        else:
            self.model = model
        self.bigrams = pickle.load(open('../authors/' + a + '_bigrams', 'rb'))
        self.wordy = pickle.load(open('../authors/' + a + '_wordy', 'rb'))
        self.class_dictionary = self.words.copy()
        print 'Successfully loaded '+self.author

    def fit(self, input_text, add_model=False):
        '''
        The actual translation occurs here:
        Methodology:
            Takes an input text, turns it to a list w/parts of speach tagging,
            then based on the part of speach, replaces certain words with
            news words that the Author would use.
        Input: String
        '''
        prt = set(string.printable)
        input_text = filter(lambda x: x in prt, input_text)
        # compare how 'wordy' Author and text are...
        wordy, input_sentences = self.tokenize_books(True, input_text)
        add_adjec = False
        # set up add
        # print wordy, self.wordy
        if int(wordy) < int(self.wordy):
            add_adjec = True
            roll = int(self.wordy) # number to check vs wordy
            last = 'Start' # Placeholder for POS
        # add to sentences for next time we rebuild our model...
        if add_model:
            sentences = pickle.load(open('../data/sentences.pickle', 'rb'))
            sentences += input_sentences
            with open('../data/sentences.pickle', 'wb') as handle:
                pickle.dump(sentences, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        threshold = self.threshold
        done = 0
        # timer...
        start = time.clock()
        words = pos_tag(word_tokenize(input_text))  # makes a list of words...
        # These simply pass thru the model
        pass_thru = ['CD',  # CD: numeral, cardinal
                     'EX',  # EX: existential there
                     'FW',  # FW: foreign word
                     'LS',  # LS: list item marker
                     'JJ',  # JJ: adjective or numeral, ordinal
                     'NNP',  # NNP: noun, proper, singular
                     'NNPS',  # NNPS: noun, proper, plural
                     'PRP',  # PRP: pronoun, personal
                     'SYM',  # SYM: symbol
                     'TO',  # TO: "to" as preposition or infinitive marker
                     'POS',
                     '$',  # $: dollar
                     '(',
                     ')',
                     ',',
                     '.',
                     ':',
                     '"'
                     ]
        # make these Basic
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
        done == 0
        count_replacements = 0
        self.lst_ret = []
        for idx, word in enumerate(words):
            if word[1] in pass_thru:
                # put it in and move on... it's proper or whatever
                self.lst_ret.append(word[0])
            else:
                # We have a word we need to replace...
                # bath it...
                clean = word[0].strip(string.punctuation).lower()
                # ...and bring it to the function
                # already simple... throw it in and move
                if clean in self.class_dictionary:
                    temp = self.class_dictionary[clean][0]
                    self.lst_ret.append(self.retain_capitalization(temp,
                                                                   word[0]))
                elif clean != '' and len(clean) > 3:
                    # not alread simply/basic, and more than 3 letters
                    # set as if we couldn't find it...
                    self.lst_ret.append(self.retain_capitalization(clean,
                                                                   word[0]))
                    # start a thread to look for it...
                    # argu = (clean, idx, word[0], word[1])
                    # t = threading.Thread(target=self.find_word, args=argu)
                    # t.daemon = True
                    # t.start()
                    abc = self.find_word(clean, word[0], word[1])
            # handle add/subtract issues..
            idx = len(self.lst_ret) - 1
            if add_adjec and word[1] in self.noun and last not in self.adj:
                if random.randint(1, roll) > wordy and len(self.lst_ret) > 1:
                    # okay, s we need to add a thing...
                    wrd = self.lst_ret[-1]
                    chk = wrd.lower().strip(string.punctuation)
                    if chk in self.bigrams:
                        new = self.bigrams[chk]
                        adj_add = new[random.randint(0, len(new)-1)]
                        insrt = self.retain_capitalization(adj_add + ' ' + wrd,
                                                           word[0])
                        self.lst_ret[-1] = insrt
                        roll = int(self.wordy) # reset this
                else:
                    # can't replace this one, but we're due for a replacement
                    roll += 1
            last = word[1]

        end = time.clock()
        if self.verbose:
            print 'Time: {:.4f}s'.format(end-start)
        txt = self.replace_punctuation(' '.join(self.lst_ret))
        txt = txt.encode('utf-8')
        txt = re.sub("\xe2\x80\x93", "-", txt)
        txt =  txt.replace('  ', ' ')
        self.basic_list = self.lst_ret
        self.basic_text = txt
        with open('../data/basic_english.pickle', 'wb') as handle:
                    pickle.dump(self.words,
                                handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
        return txt

    def find_word(self, clean, wrd, prt):
        '''
        If a word is not in the dictionary, this looks for the word given the
        most simmilar word(s) in the model loaded, determines if the simmilar
        word has been mapped already, and inserts it into the model.
        Looks for most similar word to that word in the model, then makes a
        length-threshold list of words most similar to the word before.
        Checks if these words were mapped, POS dependent.  if so, makes the
        same mapping.
        '''
        lst_ret = [clean]
        # try: # in case it fails...
        lst = []
        try:
            lst = [a[0] for a in self.model.most_similar(wrd)]
        except KeyError as e:
            if self.verbose:
                print str(e)
        if len(lst) >= 1:
            for i in xrange(self.threshold):
                c = self.model.most_similar(lst[i], topn=self.threshold)
                for d in c:
                    lst.append(d[0])
        lst = list(set(lst))
        # collect only those words in the dictionary...
        #    ...and order by simalarity
        lst = [(self.clean_word(a), self.model.similarity(wrd, a)) for a in lst
               if self.clean_word(a) in self.words]
        lst = sorted(lst, key=lambda x: -x[1])
        # lst in order
        # also, everything was mapping to 'i' for some reason...
        lst = [a[0] for a in lst if len(a[0]) > 1]
        n = 0
        done = 0
        # since we ordered by similarity, best go first
        for item in lst:
            # must match w/part of speach...
            if done == 0:
                b = pos_tag([item])[0][1]
                lst_ret.append(item)
                if b == prt:
                    # add to dictionary...based on what's there,
                    # retaining grouping info
                    done = 1
                    tmp = self.class_dictionary[item][0]
                    self.class_dictionary[clean] = [tmp, item, prt]
                    # add to lst...at the idx that cllaed it
                    idx = len(self.lst_ret) - 1
                    self.lst_ret[idx] = self.retain_capitalization(
                        self.class_dictionary[item][0], wrd)
                    if len(tmp) > 1 and len(item) > 1:
                        self.words[clean] = [tmp, item, prt]
                    lst_ret.append('*added: {}, {}*'.format(tmp, item))
            if done == 0:
                cln = clean.lower()
                a = pos_tag([cln])[0][1]
                self.class_dictionary[clean] = [cln, cln, a]
                lst_ret.append(
                    self.retain_capitalization(
                        self.class_dictionary[clean][0], wrd))
        # except:
        #     lst_ret.append('ERRORED OUT')
        #     self.class_dictionary[clean] = [clean, None, prt]
        if self.verbose:
            print lst_ret
        return lst_ret

    def get_books(self):
        '''
        Get and prepare the books...
        '''
        book_corpus = []
        to_vectorize = []
        path = '../authors/'+self.author
        prt = set(string.printable)
        for filename in os.listdir(path):
            self.book_num += 1
            if filename[-3:] == 'txt':
                # get the book
                f = open(path + '/' + filename)
                # book = Book(f.readlines())
                book = Book([filter(lambda x: x in prt, a)
                             for a in f.readlines()])
                f.close()
                # clean & remove stop words
                book.clean()
                book.remove_stop_words()
                #add to the corpus
                self.corpus.append(book)

    def get_vectorize_set(self):
        '''
        '''
        to_vectorize = []
        # get all the books
        for book in self.corpus:
            # prepare to vectorize
            [to_vectorize.append(word) for word in book.meaningful_words]
        self.vectorize_set = to_vectorize

    def vectorize_books(self):
        '''
        '''
        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        vectorizer = CountVectorizer(analyzer = "word",
                                     tokenizer = None,
                                     preprocessor = None,
                                     stop_words = None,
                                     max_features = 5000)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.
        self.data_features = vectorizer.fit_transform(self.vectorize_set)

        # Numpy arrays are easy to work with, so convert the result to an
        # array
        # self.data_features = np.toarray(self.data_features)
        # self.vocab = vectorizer.get_feature_names()
        #self.vectorizer = vectorizer

    def word_2_vec(self):
        '''
        Word to vec operation...
        '''
        # Import the built-in logging module and configure it so that Word2Vec
        # creates nice output messages
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
            level=logging.INFO)

        # Set values for various parameters
        num_features = 300    # Word vector dimensionality
        min_word_count = 5     # Minimum word count
        num_workers = 4       # Number of threads to run in parallel
        context = 10          # Context window size
        downsampling = 1e-3   # Downsample setting for frequent words

        # Initialize and train the model (this will take some time)
        from gensim.models import word2vec
        print "Training model..."
        model = word2vec.Word2Vec(self.sentences,
                                  workers=num_workers,
                                  size=num_features,
                                  min_count = min_word_count,
                                  window = context,
                                  sample = downsampling)

        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)

        # bigrams is a misnomer... it's more an adjective-gram
        phrases = Phrases(self.sentences)
        make_bigrams = Phraser(phrases)
        self.bigrams = defaultdict(list)
        fails = 0
        for sentence in make_bigrams[self.sentences]:
             for item in sentence:
                if '_' in item:
                    lst = item.split('_')
                    if len(lst) == 2:
                        try:
                            wrds = pos_tag(lst)
                            if wrds[0][1] in self.adj and wrds[1][1] in self.noun:
                                nun = wrds[1][0].lower().strip(string.punctuation)
                                adj = wrds[0][0].lower().strip(string.punctuation)
                                self.bigrams[nun].append(adj)
                        except:
                            if self.verbose:
                                print 'Error: {} -ignored'.format(lst)
        with open('../authors/' + self.author_fromatted + '_bigrams',
                  'wb') as handle:
                    pickle.dump(self.bigrams, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
        # save here to avoid having a model and no 'bigrams'
        model.save('../model/'+self.author_fromatted + '.bin')
        return model

    def tokenize_books(self, fit=False, input=None):
        '''
        Tokenize and turn into sentences.
        '''
        # Load the punkt tokenizer
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = []
        if not fit:
            for book in self.corpus:
                sentences += self.book_to_sentences(book, tokenizer)
        else:
            sentences += self.text_to_sentences(input, tokenizer)
        wordy = np.mean([len(sentence) for sentence in sentences])
        sentences = sentences
        if not fit:
            self.wordy = wordy
            self.sentences = sentences
        else:
            return wordy, sentences

    def text_to_sentences(self, text, tokenizer, remove_stopwords=False ):
        '''
        Function to split a review into parsed sentences. Returns a
        list of sentences, where each sentence is a list of words
        '''
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(text.encode("ascii","ignore")
                                           .strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append(self.book_to_wordlist(raw_sentence,
                                                  remove_stopwords))
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences

    def book_to_sentences(self, book, tokenizer, remove_stopwords=False ):
        '''
        Function to split a review into parsed sentences. Returns a
        list of sentences, where each sentence is a list of words
        '''
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(book.text.encode("ascii","ignore")
                                           .strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append(self.book_to_wordlist(raw_sentence,
                                                  remove_stopwords))
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences

    def book_to_wordlist(self, book_text, remove_stopwords=False ):
        '''
        Function to convert a document to a sequence of words,
        optionally removing stop words.  Returns a list of words.
        '''
        words = book_text.lower().split()
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        return words

    def load_sentences(self):
        '''
        '''
        sentences = []
        try:
            sentences = pickle.load(open('../data/sentences.pickle', 'rb'))
        except:
            print 'No sentences saved.'
        print 'old length = {}'.format(len(sentences))
        sentences += self.sentences
        with open('../data/sentences.pickle', 'wb') as handle:
            pickle.dump(sentences, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print 'new length = {}'.format(len(sentences))

    def retain_capitalization(self, new_word, original_word):
        '''
        Checks the original_word for capitalization, if it has it, capitalizes
        the frst letter of new_word, returns new_word.
        '''
        if original_word[0] in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            lst = list(new_word)
            lst[0] = lst[0].upper()
            new_word = ''.join(lst)
        return new_word

    def replace_punctuation(self, text):
        '''
        Tokenizing takes the punctuation as it's own item in the list.
        This takes the created string and replaces all 'end ?' with 'end?'
        '''
        text = text.replace(' .', '.')
        text = text.replace(' ?', '?')
        text = text.replace(' !', '!')
        text = text.replace(' ,', ',')
        text = text.replace(' ;', ';')
        text = text.replace(' "', '"')
        text = text.replace(" '", "'")
        text = text.replace('( ', '(')
        text = text.replace(' )', ')')
        text = text.replace('$ ', '$')
        text = text.replace(' *', '*')
        return text

    def clean_word(self, word):
        '''
        Moving a lot of functions here for speed
        '''
        check_clean = word.encode('ascii', 'replace')
        return check_clean.strip(string.punctuation).lower

if __name__ == '__main__':
    start = time.clock()
    b = '../model/300features_5min_word_count_10context.npy'
    try:
        model = gensim.models.KeyedVectors.load_word2vec_format(b, binary=True)
    except:
        model = gensim.models.Word2Vec.load(b)
    print "This took only {:.3f}s".format(time.clock()-start)
    try:
        wiki = pickle.load(open('../data/wikipedia.pickle', 'rb'))
        # articles = pickle.load(open('../data/articles_en.pickle', 'rb'))
    except:
        wiki = {}
    keys = wiki.keys()
    abc = len(keys)
    keys.sort()
    keys = keys[::-1]
    print "Starting loop..."
    for i, item in enumerate(keys):
        try:
            articles = pickle.load(open('../data/articles.pickle', 'rb'))
        except:
            articles = {}
        # check if it's been done before...
        # this is getting old
        if item not in articles:
            start = time.clock()
            # we'll lose capitalization, but whatever...
            MyText = ' '.join([' '.join(sntc) for sntc in wiki[item]])
            # translator = BasicEnglishTranslator(model, threshold=3)
            author = Author('Alexandre Dumas', model, threshold=3)
            # translator.fit(MyText)
            author.fit(MyText)
            articles[item] = [translator.basic_text, translator.basic_list,
                              translator.real_text, translator.real_list]
            with open('../data/articles.pickle', 'wb') as handle:
                pickle.dump(articles, handle, protocol=pickle.HIGHEST_PROTOCOL)
            articles = {}
            end = time.clock() - start
            # dic = translator.save_dictionary
            otr = author.words
            print "{} of {} - {}: {:.2f}s ({})".format(i, abc,
                                                           item, end,
                                                           len(otr))
