import sys
import time
import pandas as pd
import nxpd as nxpd
import random as rnd
import networkx as nx
import cPickle as pickle
import matplotlib.font_manager
from collections import Counter
import matplotlib.pyplot as plt


def make_graph_model(file=None):
    '''
    Makes a graph model from a dictionary file to show path between word and
    the word it is mapped to.
    '''
    # first, load a file...
    if file is None:
        d = pickle.load(open('../data/basic_english.pickle', 'rb'))
        # now, to make lines for nx... edgelist
        lines = []
        lines = ['{} {}'.format(key, d[key]) for key in d.keys()]
        G = nx.parse_edgelist(lines)
    else:
        d = pickle.load(open('../data/basic_english.pickle', 'rb'))
        G = pickle.load(open('../data/graph.pickle', 'rb'))
    return G, d


def make_graph(word='word'):
    '''
    Make a graph based on the words around 'word'
    '''
    d = pickle.load(open('../data/basic_english.pickle', 'rb'))
    G = pickle.load(open('../data/graph.pickle', 'rb'))
    n = 0
    done = 0
    subG = None
    print 'Starting'
    # get shortest path between target and it's word
    paths = nx.single_source_dijkstra(G, word, target=d[word][0], cutoff=None)
    # make main path
    lst = []
    lst.append(word)
    lst.append(d[word][0])
    for key in paths[1].keys():
        for a in paths[1][key]:
            lst.append(a)
    # main path made...
    # get nearest neighbors of all words in that path...
    lst_n = [a for a in lst]
    for wrd in lst_n:
        paths = nx.single_source_dijkstra(G, wrd, cutoff=1)
        for key in paths[1].keys():
            for a in paths[1][key]:
                lst.append(a)
    # make a subgraph of all the words we just found
    subG = G.subgraph(lst)
    if subG is None:
        print "I'm sorry, Dave, but {} is not in the model.".format(word)
    else:
        # color words based on inclusion
        basic_s = pd.read_csv('../data/basic_english_wordlist.csv')
        basic = list(basic_s['WORD'])
        # set up a spring layout
        set_pos = nx.spring_layout(subG)
        # color words based on if they are a key...
        print "getting nodes..."
        node_lst = subG.nodes()
        label_lst_r = {}
        label_lst_g = {}
        label_lst_b = {}
        values = list(set([d[key][0] for key in d.keys()]))
        for i, node in enumerate(node_lst):
            if node == word:
                node_lst[i] = 400
            else:
                node_lst[i] = 0
            if node in basic:
                label_lst_b[node] = node
                label_lst_g[node] = ''
                label_lst_r[node] = ''
            elif node in values:
                label_lst_b[node] = ''
                label_lst_g[node] = ''
                label_lst_r[node] = node
            else:
                label_lst_b[node] = ''
                label_lst_g[node] = node
                label_lst_r[node] = ''

        # draw network lines...
        nx.draw_networkx(subG,
                         pos=set_pos,
                         arrows=True,
                         with_labels=False,
                         node_color='y',
                         node_size=node_lst,
                         alpha=0.15,
                         font_color='k',
                         font_weight='bold')
        # add basic words
        nx.draw_networkx_labels(subG,
                                labels=label_lst_b,
                                pos=set_pos,
                                font_color='b',
                                font_weight='bold',
                                alpha=1.0)
        # add 'end' words
        nx.draw_networkx_labels(subG,
                                labels=label_lst_r,
                                pos=set_pos,
                                font_color='r',
                                font_weight='bold',
                                alpha=1.0)
        # add all other words
        nx.draw_networkx_labels(subG,
                                labels=label_lst_g,
                                pos=set_pos,
                                font_color='g',
                                font_weight='bold',
                                alpha=1.0)
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        make_graph(sys.argv[1])
    else:
        print "No word specified..."
        basic_s = pd.read_csv('../data/basic_english_wordlist.csv')
        basic = list(basic_s['WORD'])
        word = basic[rnd.randint(0, len(basic)-1)]
        print "...so we'll do, I dunno... {}".format(word)
        make_graph(word)
