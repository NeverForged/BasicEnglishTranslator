import pandas as pd
import cPickle as pickle
import nltk.data
from nltk import pos_tag, word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import gensim as gensim
import string
import re
import time
from bs4 import BeautifulSoup
import requests
import sys
import random
from collections import defaultdict

class BasicEnglishTranslator():
    '''
    Takes an input set of text and translates it to a slightly modified subset
    of the English language based on Charles K. Ogden's BASIC English.  Some
    liberties have been taken based on the nature of this translator and the
    overall goals of the ghost-writer.click project.

    PARAMETERS
    model: A word2vec model to use to search for associated words for new
        words encountered.  In command-line load, defaults to google-news.
    basic_dictionary: an input dictionary specified at instantiation, intent
        is a saved dictionary in training.  If none, loads a dictionary from a
        pickle file.  If this doesn't exist, creates one from a csv of Ogden's
        Vocab and a few word2vec models in the data directory, if available
        Dictionary is of format {'words':['words', 'word']} Where key is word
        in document, [0] is word to replace with, and [1] is where that
        connection was made in the first place.
    threshold: a number of iterations when looking for a new word

    ATTRIBUTES
    real_text: the actual text to translate, entered with fit() method
    basic_text: real_text translated to BASIC English
    real_list: List of real_text with parts of speach
    basic_list: List of basic_text with parts of speach
    class_dictionary: the used dictionary with additional translation steps
        included, if any.

    METHODS
    fit(input_text): Main function, builds the various lists and texts off
        of the input_text (string)
    '''
    def __init__(self, model, basic_dictionary=None,
                 threshold=1, verbose=False):
        '''
        Initializer.
        '''
        # model
        self.model = model
        # Dictionary
        # check valid dictionary
        self.class_dictionary = pickle.load(open('../data/basic_english.pickle',"rb"))
        # Threshold
        self.threshold = threshold
        self.save_dictionary = self.class_dictionary.copy()
        self.verbose = verbose

    def fit(self, input_text, add_model=False):
        '''
        The actual translation occurs here:
        Methodology:
            Takes an input text, turns it to a list w/parts of speach tagging,
            then based on the part of speach, replaces certain words with
            Basic English words from the dictionary.
        Input: String
        '''
        self.real_text = input_text
        threshold = self.threshold
        done = 0
        # timer...
        start = time.clock()
        prt = set(string.printable)
        input_text = filter(lambda x: x in prt, input_text)
        # add to sentences for next time we rebuild our model...
        if add_model:
            input_sentences = self.book_to_sentences(input_text)
            sentences = pickle.load(open('../data/sentences.pickle', 'rb'))
            sentences += input_sentences
            with open('../data/sentences.pickle', 'wb') as handle:
                pickle.dump(sentences, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        words = pos_tag(word_tokenize(input_text))  # makes a list of words...
        self.real_list = words

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
        end = time.clock()
        if self.verbose:
            print 'Time: {:.4f}s'.format(end-start)
        txt = self.replace_punctuation(' '.join(self.lst_ret))
        txt = txt.encode('utf-8')
        txt = re.sub("\xe2\x80\x93", "-", txt)
        self.basic_list = self.lst_ret
        self.basic_text = txt
        with open('../data/basic_english.pickle', 'wb') as handle:
                    pickle.dump(self.save_dictionary,
                                handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

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
               if self.clean_word(a) in self.save_dictionary]
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
                        self.save_dictionary[clean] = [tmp, item, prt]
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
        return check_clean.strip(string.punctuation).lower()


        # Define a function to split a book into parsed sentences
    def book_to_sentences(self, input_text, remove_stopwords=False):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw_sentences = tokenizer.tokenize(
                        input_text.encode("ascii", "replace").strip())
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


    def book_to_wordlist(self, book_text, remove_stopwords=False):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        # 3. Convert words to lower case and split themstring.decode('utf-8')
        words = book_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if w not in stops]
        #
        # 5. Return a list of words
        return words


if __name__ == '__main__':
    start = time.clock()
    # b = '../model/300features_5min_word_count_10context.npy'
    b = '../model/GoogleNews-vectors-negative300.bin'
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
    for i, item in enumerate(keys):
        try:
            articles = pickle.load(open('../data/articles.pickle', 'rb'))
        except:
            articles = {}
        # check if it's been done before...
        if item not in articles:
            start = time.clock()
            # we'll lose capitalization, but whatever...
            MyText = ' '.join([' '.join(sntc) for sntc in wiki[item]])
            translator = BasicEnglishTranslator(model,
                                                basic_dictionary=None,
                                                threshold=10)
            translator.fit(MyText)
            articles[item] = [translator.basic_text, translator.basic_list,
                              translator.real_text, translator.real_list]
            with open('../data/articles.pickle', 'wb') as handle:
                pickle.dump(articles, handle, protocol=pickle.HIGHEST_PROTOCOL)
            articles = {}
            end = time.clock() - start
            dic = translator.save_dictionary
            print "{} of {} - {}: {:.2f}s ({})".format(i, abc,
                                                       item, end, len(dic))
