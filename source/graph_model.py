import time
import gensim
import build_model as bm
import string
from nltk import pos_tag
from collections import defaultdict
import cPickle as pickle
import networkx as nx
import nxpd as nxpd
import sys
import numpy as np
import threading
from threading import Thread
import string
import unicodedata
import re

def find_sims(model, model_name):
    '''
    Given a Gensim model, creates a dictionary of lists of words that are the
    most similar words that share a part of speach.

    Once it has this list, it eliminates words that do not belong in that list
    + the original word until either (1) the list is length 2 or (2) the base
    word does not belong.

    Saves the words and 1 - cosine similarity^2
    '''
    # lock to protect dictionary...
    lock = threading.Lock()
    # try:
    ret = pickle.load(open('../data/' + model_name + '_sim_dict.pickle', 'rb'))
    # except:
    #     ret = defaultdict(list)
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
    start = time.clock()

    ylst = list('?!12345678"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r\x0b\x0c')
    keys_length = float(len(model.vocab.keys()))
    for i, word in enumerate(model.vocab.keys()):
        if word not in ret:
            if any(x in ylst for x in word):
                # weird one with punctuation, or a phrase
                # 100.0 ensures it won't show up later, but will not waste
                #  time in that list comprehension
                ret[word] = [(word, 100.0)]
            else:
                # only use the ones I want to replace
                args = (word, ret, model, make_simple,
                        lock, ylst, i, model_name)
                a = Thread(target=thread_word_search, args=args)
                a.start()
                a.join()
        if i % 5 == 0:
            per = 100.0 * i/keys_length
            if i % 20 == 0:
                print 'Get Connections {:.2f}% \  \r'.format(per),
            elif i % 15 == 0:
                print 'Get Connections {:.2f}% |  \r'.format(per),
            elif i % 10 == 0:
                print 'Get Connections {:.2f}% /  \r'.format(per),
            else:
                print 'Get Connections {:.2f}% -  \r'.format(per),
    end = time.clock()
    print 'Dictionary took {:.2f}s'.format((end - start)/60.0)
    args = (model_name + '_sim_dict.pickle', ret, lock)
    a = Thread(target=save_to_pickle, args=args)
    a.start()
    a.join()

    return ret

def thread_word_search(word, ret, model, make_simple,
                       lock, ylst, i, model_name):
    '''
    The actual word search, here so I can run in threads.
    Grabs a lock.
    Does some voodoo
    releases the lock.
    '''
    lock.acquire()
    pos = pos_tag([word.lower()])[0][1]
    if pos in make_simple:
        # it's the sort of thing we'd replace...
        try:
            # only want the non-proper nouns
            lst = [(a[0], 1 - a[1]**2) for a in
                   model.most_similar(word.lower())
                   if a[0].lower() == a[0]]
        except:
            # word wasn't in the vocab after manipulation, skip it
            lst = []
        if len(lst) > 0:
            lst = [a for a in lst if not any(x in ylst for x in a[0])]
            nlst = []
            for a in lst:
                # same part of speach only...
                if pos_tag([a[0].lower()])[0][1] == pos:
                    nlst.append(a)
            # grab top 3...
            if len(nlst) >= 3:
                nlst = nlst[:3]
            # avoid at least one opposite...
            if len(nlst) > 2:
                wlst = [a[0] for a in nlst]
                # check which don't match with the original word
                rem = model.doesnt_match(wlst + [word])
                # avoid errors
                if rem != word:
                    nlst.pop(wlst.index(rem))
            # make sure it's a list (of tuples)
            if type(nlst) != list:
                nlst = [nlst]
            ret[word] = nlst + [(word, 0.0)]
        else:
            ret[word] = [(word, 0.0)]
    else:
        ret[word] = [(word, 0.0)]
    lock.release()
    if i % 50 == 0:
        args = (model_name + '_sim_dict.pickle', ret, lock)
        a = Thread(target=save_to_pickle, args=args)
        a.start()
        a.join()

def save_to_pickle(name, info, lock):
    '''
    In a function to avoid kboard interruptions
    '''
    lock.acquire()
    with open('../data/' + name, 'wb') as handle:
            pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    lock.release()

def get_sims(model_name, model):
    missing = 0
    try:
        # grab the sims model, check if complete
        ret = pickle.load(open('../data/' + model_name + '_sim_dict.pickle',
                               'rb'))
        missing = 0
        for word in model.vocab.keys():
            if word not in ret:
                missing += 1
        if missing >= 1:
            # need to finish building the sims model
            print 'missing {} words of {}'.format(missing,
                                                  len(model.vocab.keys()))
            ret = find_sims(model, model_name)
    except:
        # need to start building the sims model
        ret = find_sims(model, model_name)
    return ret

def get_google():
    '''
    Loads google news word2vec model
    '''
    start = time.clock()
    # b = '../model/300features_5min_word_count_10context.npy'
    b = '../model/GoogleNews-vectors-negative300.bin'
    try:
        model = gensim.models.KeyedVectors.load_word2vec_format(b, binary=True)
    except:
        model = gensim.models.Word2Vec.load(b)
    model.init_sims(replace=True) # save memory
    print "This took only {:.3f}s".format(time.clock()-start)
    return model

def get_sentence_model(remake=False):
    '''
    Loads or builds a model based on the sentences in the data folder.
    '''
    start = time.clock()
    if remake:
        model = bm.build_model()
    else:
        b = '../model/300features_5min_word_count_10context.npy'
        try:
            model = gensim.models.KeyedVectors.load_word2vec_format(b,
                                                        binary=True)
        except:
            model = gensim.models.Word2Vec.load(b)
        model.init_sims(replace=True) # save memory
        print "This took only {:.3f}s".format(time.clock()-start)
    return model.wv

def make_graph_model(d):
    '''
    Makes a graph model from a dictionary file.
    '''
    # now, to make lines for nx... edgelist
    start = time.clock()
    lines = []
    G=nx.Graph()
    for key in d.keys():
        for item in d[key]:
            G.add_edge(item[0], key, weight=item[1])
    print 'G took {:.2f}s'.format(time.clock() - start)
    return G, d

def make_dictionary(a, G, input_d):
    '''
    This makes a dictionary based on an input_d which is usually created
    by some other process (basic_english dictionary maker, self.words from
    Author class, etc.)

    Takes the graph model made from the sims dictionary and searches for the
    shortest path by distance, where each edge has a value of sine similarity^2
    (aka 1 - cosine similarity^2), to each 'valid' word given by the input
    dictionary, in order from longest to shortest words.  If another word has a
    'shorter' (by edge = 1 measurement) path, it replaces it until all words
    that are in a graph with words in our dictionary have been mapped.
    '''
    vocab = input_d.keys()
    vocab.sort()
    vocab.sort(key=len, reverse=True)
    start = time.clock()
    # make a dictionary...
    # temp = nx.all_pairs_dijkstra_path_length(G, cutoff=10, weight='weight')
    try:
        paths = pickle.load(open('../data/' + a + 'temp_paths.pickle', 'rb'))
    except:
        paths = defaultdict(list)
        for i, word in enumerate(vocab):
            # temp = dictionary of source -> diction of target -> length
            try:
                temp = nx.single_source_dijkstra_path_length(G, word,
                                                            weight='weight')
            except:
                temp = {}
            tkeys = temp.keys()
            for key in tkeys:
                # compare sin^2 similarity length
                if clean_word(key) != key and clean_word(key) in tkeys:
                   # skip it, it's pos will throw us off
                    pass
                elif clean_word(key) not in input_d and len(key) > 2:
                    try:
                        length = paths[key][1]
                    except:
                        length = 10.0
                    length_n = temp[key]
                    if length > length_n:
                        paths[clean_word(key)] = (input_d[word][0],
                                      temp[key],
                                      input_d[word][1])
            if i % 25 == 0:
                per = 100.0*i/float(len(vocab))
                print 'Pathfinder({}):  {:.2f}%'.format(a, per)
        print 'Paths Found, Took {:.2f}s'.format(time.clock() - start)
        with open('../data/' + a + 'temp_paths.pickle', 'wb') as handle:
                pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # now set our dictionary
    start = time.clock()
    for i, key in enumerate(paths.keys()):
        print '    Dictionary: \r',
        try:
            pos = pos_tag([paths[key][0]])[0][1]

            # i still don't trust the pos tagging
            # if pos == pos_tag([paths[key][0]]):
            input_d[key.lower()] = [paths[key][0].lower(),
                                    paths[key][2].lower(),
                                    pos]
        except:
            print key, paths[key]
        if i % 25  == 0:
            per = 100.0*i/float(len(paths))
            print '    Dictionary: {:.0f}% \r'.format(per),
    print 'Dictionary Made, Took {:.2f}s'.format(time.clock() - start)
    if a == 'Basic':
        with open('../data/temp_basic_english.pickle', 'wb') as handle:
                pickle.dump(input_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('../data/temp_' + a + '.pickle', 'wb') as handle:
                pickle.dump(input_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return input_d


def set_words(author, lock):
    '''
    Using threading to make multiple Author's at once.
    '''
    # lock.acquire()
    # first, need to grab the dictionary of the Author...
    a = author.encode('ascii', 'replace')
    a = a.lower().strip().replace(' ','_')
    missing = 0
    if author == 'Basic':
        newd = pickle.load(open('../data/basic_english - Copy.pickle', 'rb'))
    else:
        try:
            newd = pickle.load(open('../authors/' + a + '_words', 'rb'))
        except:
            Author(author)
            newd = pickle.load(open('../authors/' + a + '_words', 'rb'))

    keys = newd.keys()
    thed = make_dictionary(a, G, newd)
    for key in keys:
        try:
            alpha = thed[key]
        except:
            missing += 1
    print '{} missing {} entries'.format(author, missing)


    # save it...
    if missing == 0:
        if author == 'Basic':
            with open('../data/basic_english.pickle', 'wb') as handle:
                    pickle.dump(thed, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('../authors/' + a +'_words', 'wb') as handle:
                    pickle.dump(thed, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            with open('../authors/' + a +'_words_og', 'wb') as handle:
                    pickle.dump(newd, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
    # lock.release()

def clean_word(word):
    '''
    Cleans all the non-ascii stuff
    '''
    try:
        word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore')
    except:
        word = word
    prt = set(string.printable)
    word = filter(lambda x: x in prt, word)
    word = word.encode('utf-8')
    word = re.sub("\xe2\x80\x93", "-", word)
    return word.lower()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'google':
            model = get_google()
    # else:
    #     model = get_sentence_model()
    d = get_sims('test', model)
    G, d = make_graph_model(d)
    #
    # del model
    lock = threading.Lock()
    lst = ['Basic']
    # lets do this...
    threads = []
    for author in lst:
        args = (author, lock)
        threads.append(Thread(target=set_words, args=args))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
