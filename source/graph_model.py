import time
import gensim
import build_model as bm
import string
from nltk import pos_tag
from collections import defaultdict
import cPickle as pickle
import networkx as nx
import nxpd as nxpd

def find_sims(model):
    '''
    Given a Gensim model, creates a dictionary of lists of words that are the
    most similar words that share a part of speach (len 0-10)
    '''
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
    ret = defaultdict(list)
    ylst = list('?!12345678"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r\x0b\x0c')
    for i, word in enumerate(model.vocab.keys()):
        if any(x in ylst for x in word):
            # weird one with punctuation, or a phrase
            pass
        else:
            # only use the ones I want to replace
            pos = pos_tag([word.title()])[0][1]
            if pos in make_simple:
                lst = []
                lst = [a[0].lower() for a in model.most_similar(word.lower())
                       if a[0].lower() == a[0]]
                lst = [a for a in lst if not any(x in ylst for x in a)]
                nlst = []
                for a in lst:
                    if pos_tag([a])[0][1] == pos:
                        nlst.append(a)
                if len(nlst) > 2:
                    for i in xrange(int(len(nlst)/2)):
                        rem = model.doesnt_match(nlst + [word])
                        if rem == word:
                            pass
                        else:
                            nlst.pop(nlst.index(rem))
                ret[word] = nlst
            if i % 10 == 0:
                per = 100.0 * i/float(len(model.vocab.keys()))
                print 'Get Connections {:.1f}%  \r'.format(per),

    end = time.clock()
    print 'Dictionary took {:.2f}s'.format((end - start)/60.0)

    return ret

def get_sims(model_name, model):
    try:
        ret = pickle.load(open('../data/' + model_name + '_sim_dict.pickle',
                               'rb'))
    except:
        ret = find_sims(model)
        with open('../data/' + model_name + '_sim_dict.pickle', 'wb') as handle:
                pickle.dump(ret, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
    lines = []
    for key in d.keys():
        for item in d[key]:
            lines.append('{} {}'.format(key, item))
    G = nx.parse_edgelist(lines)
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
    lst = [A for A in nx.connected_components(G) if any(x in vocab for x in A)]
    # make only the subgraphs that connect to at least one of our words
    Graphs = []
    for A in lst:
        Graphs.append(G.subgraph(A))

    start = time.clock()
    paths = defaultdict(list)
    # make a dictionary...
    for i, word in enumerate(vocab):
        for graph in Graphs:
            # this should only come up once per
            if word in graph.nodes():
                # each node...
                for point in graph.nodes():
                    # find path, replace if needed
                    path = nx.shortest_path(G, point, word)
                    if len(paths[point]) < len(path):
                        paths[point] = path
        if i % 50 == 0:
            per = 100.0*i/float(len(vocab))
            print '    Pathfinder: {:.0f}% \r'.format(per),
    print 'Paths Found, Took {:.2f}s'.format(time.clock() - start)

    # now set our dictionary
    start = time.clock()
    for i, key in enumerate(paths.keys()):
        pos = pos_tag([paths[key][-1]])
        # i still don't trust the pos tagging
        if pos == pos_tag([paths[key][0]]):
            input_d[key] = [paths[key][-1], paths[key][1], pos]
        if i % 50 == 0:
            per = 100.0*i/float(len(paths))
            print '    Dictionary: {:.0f}% \r'.format(per),
    print 'Dictionary Made, Took {:.2f}s'.format(time.clock() - start)
    with open('../data/temp_dict.pickle', 'wb') as handle:
            pickle.dump(input_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return input_d


if __name__ == '__main__':
    model = get_sentence_model()
    d = get_sims('test', model)
    G, d = make_graph_model(d)
    print d.keys()
    newd = pickle.load(open('../data/basic_english - Copy small.pickle',
                           'rb'))
    make_dictionary(G, newd)
