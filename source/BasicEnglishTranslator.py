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
from math import sqrt


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
                      is a saved dictionary in training.  If none, loads a
                      dictionary from a pickle file.  If this doesn't exist,
                      creates one from a csv of Ogden's Vocab and a few
                      word2vec models in the data directory, if available
                      Dictionary is of format {'words':['words', 'word']}
                      Where key is word in document, [0] is word to replace
                      with, and [1] is where that connection was made in the
                      first place.
    threshold: a time, in seconds, that it will spend searching for a new
               word before giving up.

    ATTRIBUTES
    real_text: the actual text to translate, entered with fit() method
    basic_text: real_text translated to BASIC English
    real_list: List of real_text with parts of speach
    basic_list: List of basic_text with parts of speach
    class_dictionary: the used dictionary with additional translation steps
                      included, if any
    '''
    def __init__(self, model, basic_dictionary=None, threshold=0.25):
        '''
        Initializer.  Takes the real_text as an input string.
        '''
        # model
        if type(model) == gensim.models.keyedvectors.KeyedVectors:
            self.model = model
        else:
            raise ValueError('The model specified is not a valid model of \
                              type <gensim.models.keyedvectors.KeyedVectors>')
        # Dictionary
        # check valid dictionary
        if ((type(basic_dictionary) == dict) and
                (len(basic_dictionary.values()[0]) == 3)):
            self.class_dictionary = basic_dictionary
        else:  # load our vetted dictionary...
            try:
                self.class_dictionary = self.load_dictionary()
            except:
                raise KeyError('No keys since there is no dictionary.')
        # Threshold
        self.threshold = threshold
        self.save_dictionary = self.class_dictionary.copy()

    def fit(self, input_text):
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
        input_text = input_text.replace('—', ' - ').replace("’", " ' ")
        input_text = ''.join([a if ord(a) < 128 else ''
                              for a in list(input_text)])
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
        lst_ret = []
        for word in words:
            if word[1] in pass_thru:
                # put it in and move on... it's proper or whatever
                lst_ret.append(word[0])
            else:
                # We have a word we need to replace...
                # bath it...
                clean = word[0].strip(string.punctuation).lower()
                # ...and bring it to the function
                # already simple... throw it in and move
                if clean in self.class_dictionary:
                    temp = self.class_dictionary[clean][0]
                    lst_ret.append(self.retain_capitalization(temp, word[0]))
                elif clean != '':  # not alread simply/basic..
                    try:  # in case it fails...
                        start_this = time.clock()
                        lst = list(set(self.model.most_similar(clean)))
                        done = 0
                        n = 0
                        while done == 0:
                            check = list(lst)[n][0]
                            n += 1
                            check_clean = check.strip(string.punctuation)
                            ccln = check_clean.lower()
                            b = pos_tag([ccln])[0][1]
                            if ccln in self.class_dictionary and b == word[1]:
                                done = 1
                                # add to dictionary...based on what's there,
                                # retaining grouping info
                                ccln = check_clean.lower()
                                tmp = self.class_dictionary[check_clean][0]
                                a = pos_tag([tmp])[0][1]
                                self.class_dictionary[clean] = [tmp, ccln, a]
                                # add to lst
                                lst_ret.append(
                                    self.retain_capitalization(
                                        self.class_dictionary[clean][0],
                                        word[0]))
                                self.save_dictionary[clean] = [tmp, ccln, a]
                            else:
                                # add all similar words to that to the lst
                                if time.clock() - start_this < threshold:
                                    [lst.append(a) for a in
                                     self.model.most_similar(check, topn=3)
                                     if a not in lst]
                                else:  # timeout!
                                    done = 1
                                    cl = clean.lower()
                                    a = pos_tag([cln])[0][1]
                                    self.class_dictionary[clean] = [cl, cl, a]

                                    lst_ret.append(
                                        self.retain_capitalization(
                                            self.class_dictionary[clean][0],
                                            word[0]))
                    except:
                        a = word[0]
                        lst_ret.append(self.retain_capitalization(a, a))
                        temp = word[0].lower()
                        self.class_dictionary[temp] = [temp, None, word[1]]
        end = time.clock()
        print 'Time: {:.2f}s'.format(end-start)
        txt = self.replace_punctuation(' '.join(lst_ret))
        txt = txt.encode('utf-8')
        txt = re.sub("\xe2\x80\x93", "-", txt)
        self.basic_list = lst_ret
        self.basic_text = txt

    def load_dictionary(self):
        '''Loads the dictionary...'''
        return pickle.load(open('../data/basic_english.pickle', "rb"))

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


# external to class for the sake of saving model stuff
# Define a function to split a book into parsed sentences
def book_to_sentences(input_text, tokenizer, remove_stopwords=False):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(
                    input_text.encode("ascii", "replace").strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(book_to_wordlist(raw_sentence, remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def book_to_wordlist(book_text, remove_stopwords=False):
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
    start =  time.clock()
    articles = {}
    lst = [u'/wiki/Vocabulary', u'/wiki/Democracy', u'/wiki/Execution',
           u'/wiki/Architecture', u'/wiki/Communication', u'/wiki/Electronics',
           u'/wiki/Engineering', u'/wiki/Farming', u'/wiki/Health',
           u'/wiki/Industry', u'/wiki/Medicine', u'/wiki/Transport',
           u'/wiki/Weather', u'/wiki/Anthropology', u'/wiki/Archaeology',
           u'/wiki/Geography', u'/wiki/Education', u'/wiki/History',
           u'/wiki/Language', u'/wiki/Philosophy', u'/wiki/Psychology',
           u'/wiki/Sociology', u'/wiki/Teaching',  u'/wiki/Animation',
           u'/wiki/Art', u'/wiki/Book', u'/wiki/Cooking',
           u'/wiki/Custom', u'/wiki/Culture', u'/wiki/Dance',
           u'/wiki/Family', u'/wiki/Game',  u'/wiki/Gardening',
           u'/wiki/Leisure', u'/wiki/Movie', u'/wiki/Music',
           u'/wiki/Radio',  u'/wiki/Sport', u'/wiki/Theatre',
           u'/wiki/Travel', u'/wiki/Television', u'/wiki/Algebra',
           u'/wiki/Astronomy', u'/wiki/Biology', u'/wiki/Chemistry',
           u'/wiki/Ecology', u'/wiki/Geometry', u'/wiki/Mathematics',
           u'/wiki/Physics', u'/wiki/Statistics', u'/wiki/Zoology',
           u'/wiki/Copyright', u'/wiki/Economics', u'/wiki/Government',
           u'/wiki/Law', u'/wiki/Military', u'/wiki/Politics',
           u'/wiki/Trade', u'/wiki/Atheism', u'/wiki/Buddhism',
           u'/wiki/Christianity', u'/wiki/Esotericism', u'/wiki/Hinduism',
           u'/wiki/Islam', u'/wiki/Jainism', u'/wiki/Judaism',
           u'/wiki/Mythology', u'/wiki/Paganism', u'/wiki/Sect',
           u'/wiki/Sikhism', u'/wiki/Taoism', u'/wiki/Theology',
           u'/wiki/Horse', u'/wiki/France', u'/wiki/French_Revolution',
           u'/wiki/Sword', u'/wiki/Gun', u'/wiki/War']
    model = gensim.models.KeyedVectors.load_word2vec_format(
            '../model/GoogleNews-vectors-negative300.bin', binary=True)
    try:
        save_dic = pickle.load(open('../data/training.pickle', 'rb'))
    except:
        save_dic = None
    sentences = []
    print 'this took {}s'.format(time.clock()-start)
    for i, item in enumerate(lst):
        start = time.clock()
        translator = BasicEnglishTranslator(model, basic_dictionary=save_dic)
        r = requests.get('https://simple.wikipedia.org'+item)
        soup = BeautifulSoup(r.content, 'html.parser')
        tags = soup.find_all('p')
        MyText = '\n'.join([tag.get_text() for tag in tags])

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
        features = vectorizer.fit_transform([MyText])
        # Numpy arrays are easy to work with, so convert the result to an
        # array
        features = features.toarray()
        vocab = vectorizer.get_feature_names()
        # Load the punkt tokenizer
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # Load in sentences
        MyText = MyText.encode('ascii', 'replace')
        try:
            sentences = pickle.load(open('../data/sentences.pickle', 'rb'))
        except:
            print 'No sentences saved.'
        sentences += book_to_sentences(MyText, tokenizer)
        with open('../data/sentences.pickle', 'wb') as handle:
            pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sentences = []
        translator.fit(MyText)
        try:
            aticles = pickle.load(open('../data/sentences.pickle', 'rb'))
        except:
            print 'eh'
        articles[item.replace('/wiki/', '')] = [translator.basic_text,
                                                translator.basic_list,
                                                translator.real_text,
                                                translator.real_list]

        with open('../data/articles.pickle', 'wb') as handle:
            pickle.dump(articles, handle, protocol=pickle.HIGHEST_PROTOCOL)
        articles = {}
        save_dic = translator.save_dictionary
        with open('../data/training.pickle', 'wb') as handle:
            pickle.dump(save_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        end = time.clock()-start
        print "{} of {}: {}s".format(i, len(lst), end)
