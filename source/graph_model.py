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


def find_sims(model):
    '''
    Given a Gensim model, creates a dictionary of lists of words that are the
    most similar words that share a part of speach (len 0-10)
    '''
    try:
        ret = pickle.load(open('../data/' + model_name + '_sim_dict.pickle',
                               'rb'))
    except:
        ret = defaultdict(list)
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
        if word not in dic:
            if any(x in ylst for x in word):
                # weird one with punctuation, or a phrase
                pass
            else:
                # only use the ones I want to replace
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
                        try:
                            # grab top 3....
                            nlst = nlst[:3]
                        except:
                            uneeded_step = 0
                        # avoid at least one opposite...
                        if len(nlst) > 2:
                            wlst = [a[0] for a in nlst]
                            # check which don't match with the original word
                            rem = model.doesnt_match(wlst + [word])
                            if rem == word:
                                pass
                            else:
                                nlst.pop(wlst.index(rem))
                        ret[word] = nlst + [(word, 0.0)]
        if i % 50 == 0:
            per = 100.0 * i/keys_length
            if i % 200 == 0:
                print 'Get Connections {:.2f}% \  \r'.format(per),
            elif i % 150 == 0:
                print 'Get Connections {:.2f}% |  \r'.format(per),
            elif i % 100 == 0:
                print 'Get Connections {:.2f}% /  \r'.format(per),
            else:
                print 'Get Connections {:.2f}% -  \r'.format(per),
            with open('../data/' + model_name + '_sim_dict.pickle',
                      'wb') as handle:
                    pickle.dump(ret, handle, protocol=pickle.HIGHEST_PROTOCOL)
    end = time.clock()
    print 'Dictionary took {:.2f}s'.format((end - start)/60.0)

    return ret

def get_sims(model_name, model):
        ret = find_sims(model)
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

def make_dictionary(G, input_d):
    '''
    This makes a dictionary based on an input_d which is usually created
    by some other process (basic_english dictionary maker, self.words from
    Author class, etc.)
    '''
    vocab = input_d.keys()
    vocab.sort()
    vocab.sort(key=len, reverse=True)

    start = time.clock()
    paths = defaultdict(list)
    # make a dictionary...
    for i, word in enumerate(vocab):
        try:
            temp = nx.shortest_path(G, target=word, weight='weight')
        except:
            temp = {word:word}
        for key in temp.keys():
            length = len(paths[key])
            length_n = len(temp[key])
            if length == 0 or length < length_n:
                paths[key] = temp[key]
        if i % 25 == 0:
            per = 100.0*i/float(len(vocab))
            print '    Pathfinder: {:.0f}% \r'.format(per),
    print 'Paths Found, Took {:.2f}s'.format(time.clock() - start)
    with open('../data/temp_paths.pickle', 'wb') as handle:
            pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # now set our dictionary
    start = time.clock()
    for i, key in enumerate(paths.keys()):
        pos = pos_tag([paths[key][-1]])[0][1]
        # i still don't trust the pos tagging
        # if pos == pos_tag([paths[key][0]]):
        if type(paths[key]) != list:
            input_d[key] = [paths[key], paths[key], pos]
        if len(paths[key]) >= 2:
            input_d[key] = [paths[key][-1], paths[key][1], pos]
        if i % 25  == 0:
            per = 100.0*i/float(len(paths))
            print '    Dictionary: {:.0f}% \r'.format(per),
    print 'Dictionary Made, Took {:.2f}s'.format(time.clock() - start)
    return input_d

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'google':
            model = get_google()
    else:
        model = get_sentence_model()
    d = get_sims('test', model)
    print d['writ']
    print d['spoon']
    # d = {'word':['words', 'wordy', 'worded'], 'thing':['word', 'wizzy', 'wick', 'foo'], 'foo':['bar']}
    G, d = make_graph_model(d)
    # print G.nodes()
    newd = pickle.load(open('../data/basic_english - Copy small.pickle',
                           'rb'))
    keys = newd.keys()
    del model
    # newd = {'foo':['foo', 'foo', 'NN'], 'word':['word', 'word', 'NN']}
    thed = make_dictionary(G, newd)
    # print thed
    missing= 0
    for key in keys:
        try:
            a = thed[key]
        except:
            missing += 1
    print 'missing {} entries'.format(missing)
