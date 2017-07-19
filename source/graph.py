import sys
import networkx as nx
import nxpd as nxpd
import matplotlib.pyplot as plt
import matplotlib.font_manager
from collections import Counter
import cPickle as pickle
import time
import pandas as pd

def make_graph_model(file=None):
    '''
    Makes a graph model from a dictionary file.
    '''
    # first, load a file...
    if file == None:
        d = pickle.load(open('../data/basic_english.pickle', 'rb'))
    else:
        d = pickle.load(open(file, 'rb'))

    # now, to make lines for nx... edgelist
    lines = ['{} {}'.format(key, d[key][1]) for key in d.keys()]
    G = nx.parse_edgelist(lines)
    return G, d


def make_graph(word='word'):
    '''
    '''
    try:
        G, d = make_graph_model(None)
        tup = ()
        lst = [list(a) for a in nx.connected_components(G)]
        for i, a in enumerate(lst):
            if word in a:
                 tup = (i, a)
        subG = G.subgraph(tup[1])

        basic_s = pd.read_csv('../data/basic_english_wordlist.csv')
        basic = list(basic_s['WORD'])
        # set up a spring layout
        set_pos = nx.spring_layout(subG)

        # color words based on if they are a key...
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

        nx.draw_networkx(subG,
                         pos=set_pos,
                         arrows=True,
                         with_labels=False,
                         node_color='y',
                         node_size=node_lst,
                         alpha=0.15,
                         font_color='k',
                         font_weight='bold')
        nx.draw_networkx_labels(subG,
                                labels=label_lst_b,
                                pos=set_pos,
                                font_color='b',
                                font_weight='bold',
                                alpha=1.0)
        nx.draw_networkx_labels(subG,
                                labels=label_lst_r,
                                pos=set_pos,
                                font_color='r',
                                font_weight='bold',
                                alpha=1.0)
        nx.draw_networkx_labels(subG,
                                labels=label_lst_g,
                                pos=set_pos,
                                font_color='g',
                                font_weight='bold',
                                alpha=1.0)
        # nx.draw_graphviz(G)
        # plt.legend()
        plt.show()
    except:
        print 'Looks like "{}" is not in the dictionary!'.format(word)

def green_words():
    '''
    Finds only 'green' words... and does the thing where it shows you the most
    connected green words...
    '''
    c = pickle.load(open('../data/basic_english - Copy.pickle', 'rb'))
    G, d = make_graph_model()

    original = {}
    for item in c.keys():
        original[item] = item

    # items that were added by the translator
    new = {}
    for item in d.keys():
        if item not in original:
            new[item] = item

    # get degree centrality
    deg_cen = nx.eigenvector_centrality(G)
    cntr = Counter(deg_cen)
    mc = cntr.most_common(1000)
    lst = []
    for item in mc:
        if item[0] in new:
            lst.append(item[0])
    if len(lst) == 0:
        lst = new.keys()
    print lst
    make_graph(lst[0])



if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '@':
            green_words()
        else:
            make_graph(sys.argv[1])
    else:
        G, d = make_graph_model(file=None)
        deg_cen = nx.degree_centrality(G)
        cntr = Counter(deg_cen)
        mc = cntr.most_common(100)
        basic_s = pd.read_csv('../data/basic_english_wordlist.csv')
        basic = list(basic_s['WORD'])
        lst = []
        for item in mc:
            if item[0] in basic:
                lst.append(item[0])
        print lst
        make_graph(lst[0])
