import pandas as pd
import cPickle as pickle
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import gensim as gensim
import string
import re
import nltk.data
import time


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
    def __init__(self, model, basic_dictionary=None,
                 threshold=0.5):
        '''
        Initializer.  Takes the real_text as an input string.
        '''
        # Text
        self.real_text = real_text
        # model
        if type(model) == gensim.models.keyedvectors.KeyedVectors:
            self.model = model
        else:
            raise ValueError('The model specified is not a valid model of \
                              type <gensim.models.keyedvectors.KeyedVectors>')
        # Dictionary
        # check valid dictionary
        if ((type(basic_dictionary) == dict) and
                (len(basic_dictionary.values[0]) == 2)):
            self.class_dictionary = basic_dictionary
        else:  # load our vetted dictionary...
            try:
                self.class_dictionary = load_dictionary()
            except:
                self.class_dictionary = make_new_dictionary()
        # Threshold
        self.threshold = threshold

    def fit(real_text):
        '''
        The actual translation occurs here:

        Methodology:
            Takes an input text, turns it to a list w/parts of speach tagging,
            then based on the part of speach, replaces certain words with
            Basic English words from the dictionary.

        Input: String
        '''
        if threshold == 0:
            threshold = 10.0/sqrt(len(input_text))
        done = 0
        # timer...
        start = time.clock()
        input_text = input_text.replace('—', ' - ').replace("’", " ' ")
        input_text = ''.join([a if ord(a) < 128 else ''
                              for a in list(input_text)])
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
                if clean in self.class_dictionary.keys():
                    temp = self.class_dictionary[clean][0]
                    lst_ret.append(retain_capitalization(temp, word[0]))
                elif clean != '':  # not alread simply/basic...
                    start_this = time.clock()  # timing for testing
                    try:  # in case it fails...
                        lst = list(set(Google_model.most_similar(clean)))
                        done = 0
                        n = 0
                        while done == 0:
                            check = list(lst)[n][0]
                            n += 1
                            check_clean = check.strip(string.punctuation)
                            check_clean = check_clean.lower()
                            if check_clean in self.class_dictionary.keys():
                                done = 1
                                # add to dictionary...based on what's there,
                                # retaining grouping info
                                ccln = check_clean.lower()
                                tmp = self.class_dictionary[check_clean][0]
                                self.class_dictionary[clean] = [tmp, ccln]
                                # add to lst
                                lst_ret.append(
                                    retain_capitalization(
                                        self.class_dictionary[clean][0],
                                        word[0]))
                            else:
                                # add all similar words to that to the lst
                                if time.clock() - start_this < threshold:
                                    [lst.append(a) for a in
                                     model.most_similar(check, topn=3)
                                     if a not in lst]
                                else:  # timeout!
                                    done = 1
                                    cln = clean.lower()
                                    self.class_dictionary[clean] = [cln, cln]
                                    lst_ret.append(
                                        retain_capitalization(
                                            self.class_dictionary[clean][0],
                                            word[0]))
                    except:
                        lst_ret.append(retain_capitalization(word[0], word[0]))
                        temp = word[0].lower()
                        self.class_dictionary[temp] = [temp, None]
        end = time.clock()
        print 'Time: {:.2f}s'.format(end-start)
        txt = replace_punctuation(' '.join(lst_ret))
        txt = txt.encode('utf-8')
        txt = re.sub("\xe2\x80\x93", "-", txt)
        self.basic_list = lst_ret
        self.basic_text = txt

    def load_dictionary():
        '''Loads the dictionary...'''
        return pickle.load(open('data/basic_english.pickle', "rb"))

    def make_new_dictionary():
        '''
        This means the data file associated with this particular instance does
        not have a 'data/basic_english.pickle' file associated with it, and
        we need to make one.
        '''
        basic_english = get_basic_english()
        st = LancasterStemmer()
        stem_gn = [st.stem(key) for key in self.model.vocab.keys()]
        stem_se = [st.stem(word) for word in basic_english]
        my_dict = {}
        threshold = 0.2  # much smaller and it grabs weird things
        for sim_in in xrange(len(basic_english)-1, 0, -1):
            print
            print basic_english[sim_in]
            print '*'*17
            indices = [i for i, s in enumerate(stem_gn)
                       if stem_se[sim_in] == s]
            check = [i for i, s in enumerate(vocab_google)
                     if basic_english[sim_in] == s]
            # check, indices
            if len(check) > 0:
                for index in indices:
                    if (Google_model.similarity(basic_english[sim_in],
                                                vocab_google[index]) >=
                            threshold):
                        print '{} -> {}'.format(vocab_google[index],
                                                Google_model.similarity(
                                                basic_english[sim_in],
                                                vocab_google[index]))
                        my_dict[vocab_google[index].lower()] = \
                               [vocab_google[index].lower(),
                                basic_english[sim_in].lower()]
            # add itself last... to overwrite issues w/above
            my_dict[basic_english[sim_in].lower()] = \
                   [basic_english[sim_in].lower(),
                    basic_english[sim_in].lower()]

            my_dict['i'] = ['i', 'i']  # add 'I
            basic_english.append('i')
            for word in basic_english:
                wordy = word
                if len(word) <= 1:
                    wordy = word+"'"
                for con in contractions:
                    if wordy.lower() in con.lower()[0:len(wordy)]:
                        my_dict[con.lower()] = [con.lower(), word.lower()]
            # these need adding/fixing
            my_dict["am"] = ['am', 'am']
            my_dict["a"] = ['a', 'a']
            return my_dict

    def get_basic_english():
        '''
        Creates a list of basic english words from a csv file containing
        Ogden's 850 basic english words.
        '''
        # Convert basic english words to a list
        basic_english_df = pd.read_csv('data/basic_english_wordlist.csv')
        basic_english = [a for a in basic_english_df['WORD']]
        # add the various conjugations of 'to be' and 'a'
        basic_english.append('an')
        basic_english.append('is')
        basic_english.append('was')
        basic_english.append('are')
        basic_english.append('were')
        basic_english.append('they')
        # 'I' causes weird issues...
        basic_english[basic_english.index('I')] = 'big'
        basic_english.append('she')
        basic_english.append('hers')
        basic_english.append('his')
        basic_english.append('my')
        basic_english.append('him')
        basic_english.append('her')
        basic_english.append('your')
        basic_english.append('their')
        basic_english.append('might')
        basic_english.append('must')
        basic_english.append('can')
        basic_english.append('did')
        basic_english.append('could')
        basic_english.append('should')
        basic_english.append('would')
        basic_english.append('that')
        basic_english.append('what')
        basic_english.append('we')
        basic_english.append('small')
        basic_english[basic_english.index('colour')] = 'color'
        return basic_english

    def retain_capitalization(new_word, original_word):
        '''
        Checks the original_word for capitalization, if it has it, capitalizes
        the frst letter of new_word, returns new_word.
        '''
        if original_word[0] in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            lst = list(new_word)
            lst[0] = lst[0].upper()
            new_word = ''.join(lst)
        return new_word

    def replace_punctuation(text):
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


if __name__ == '__main__':
