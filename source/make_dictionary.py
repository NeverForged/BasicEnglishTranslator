import gensim
import requests
import nltk.data
import cPickle as pickle
from bs4 import BeautifulSoup
from nltk import pos_tag, word_tokenize
from nltk.stem.lancaster import LancasterStemmer


def get_basic_english():
    '''
    Grabs basic english words from the website...
    '''
    r = requests.get('http://ogden.basic-english.org/word2000.html')
    soup = BeautifulSoup(r.content, 'html.parser')
    tags = soup.find_all('p')
    txt = ' '.join([tag.get_text() for tag in tags])
    txt = txt[txt.index('Basic:'):txt.index('zookeeper')+9]
    txt = txt.replace(',', '').replace('  ', ' ').replace("'", '')
    txt = txt.replace('.', '').replace('-', ' ').replace('/', ' ')
    txt = txt.replace('(', '').replace(')', '')
    txt = txt.replace('[', '').replace(']', '')
    lst = txt.split()
    lst = [a for a in lst if a.lower() == a and a[0] not in '(1234567890' and
           len(a) > 1]
    return lst


def get_og_english():
    '''
    Scrapes the original words...
    '''
    r = requests.get('http://ogden.basic-english.org/words.html')
    soup = BeautifulSoup(r.content, 'html.parser')
    tags = soup.find_all('p')
    txt = ' '.join([tag.get_text() for tag in tags])
    txt = txt[:txt.index('wrong')+5]
    txt = txt.replace(',', '').replace('  ', ' ').replace("'", '')
    txt = txt.replace('.', '').replace('-', ' ').replace('/', ' ')
    txt = txt.replace('(', '').replace(')', '')
    txt = txt.replace('[', '').replace(']', '')
    lst = txt.split()
    lst = [a for a in lst if a.lower() == a and a[0] not in '(1234567890' and
           len(a) > 1]
    return lst


def get_expanded_english():
    '''
    Grabs the international and business case words.
    '''
    r = requests.get('http://ogden.basic-english.org/intlword.html')
    soup = BeautifulSoup(r.content, 'html.parser')
    tags = soup.find_all('p')
    txt = ' '.join([tag.get_text() for tag in tags])
    txt = txt[txt.index('alcohol'):9299]
    txt = txt + ' international national write wrote'
    txt = txt.replace(',', '').replace('  ', ' ').replace("'", '')
    txt = txt.replace('.', '').replace('-', ' ').replace('/', ' ')
    txt = txt.replace('(', '').replace(')', '')
    txt = txt.replace('[', '').replace(']', '')
    txt = txt + ' i a an'
    lst = txt.split()
    lst = [a for a in lst if a.lower() == a and a[0] not in '(1234567890' and
           len(a) > 1]
    return lst


def get_second_level_english():
    '''
    Grabs a few more words
    '''
    r = requests.get('http://ogden.basic-english.org/wordalpn.html')
    soup = BeautifulSoup(r.content, 'html.parser')
    tags = soup.find_all('p')
    txt = ' '.join([tag.get_text() for tag in tags])
    txt = txt[txt.index('absence'):txt.index('Back to') - 1]
    txt = txt.replace(',', '').replace('  ', ' ').replace("'", '')
    txt = txt.replace('.', '').replace('-', ' ').replace('/', ' ')
    txt = txt.replace('(', '').replace(')', '')
    txt = txt.replace('[', '').replace(']', '')
    txt = txt + ' buy sell yard inch amp watt jule newton volt hertz'
    txt = txt + 'meter liter gram second hour day minute information data'
    txt = txt + 'men women'
    lst = txt.split()
    lst = [a for a in lst if a.lower() == a and a[0] not in '(1234567890' and
           len(a) > 1]
    return lst


def get_compound_words():
    '''
    Just what it says on the tin.
    http://ogden.basic-english.org/wordacom.html
    '''
    r = requests.get('http://ogden.basic-english.org/wordacom.html')
    soup = BeautifulSoup(r.content, 'html.parser')
    tags = soup.find_all('ul')
    txt = ' '.join([tag.get_text() for tag in tags])
    txt = txt[txt.index('aftereffect'):txt.index('These are') - 1]
    txt = txt.replace(',', '').replace('  ', ' ').replace("'", '')
    txt = txt.replace('.', '').replace('-', ' ').replace('/', ' ')
    txt = txt.replace('(', '').replace(')', '')
    txt = txt.replace('[', '').replace(']', '')
    lst = txt.split()
    lst = [a for a in lst if a.lower() == a and a[0] not in '(1234567890' and
           len(a) > 1]
    return lst


def make_dict():
    '''
    Attempting to capture endings, etc.
    '''
    lst = get_expanded_english() + get_og_english()
    lst = list(set(lst))
    lst.sort()

    with open('../data/basic_english_wordlist.csv', 'w') as f:
        f.write("WORD"+'\n')
        for word in lst:
            word = word.strip()
            word = word.replace('[', '').replace(']', '').replace('.', '')
            word = word.replace(';', '').replace(':', '').replace(',', '')
            f.write(word+'\n')
    lst = lst + get_basic_english()
    lst = lst + get_second_level_english() + get_compound_words()
    lst = list(set(lst))
    lst.sort()
    lst.sort(key=len, reverse=True)
    lst = [a for a in lst if len(a) > 1]
    b = '../model/GoogleNews-vectors-negative300.bin'
    try:
        model = gensim.models.KeyedVectors.load_word2vec_format(b, binary=True)
    except:
        model = gensim.models.Word2Vec.load(b)
    prefixes = ['',
                'un',  # Negative adjectives are formed with "UN"-
                'non',  # "Non-" is common and is considered to be neutral
                ]
    suffixes = ['',
                's',  # Plurals are formed with a trailing "S".
                'es',  # The normal exceptions of standard English also apply,
                'ies',  # notably "ES" and "IES".
                'ves',  # wives, elves
                'er',  # There are four derivatives for the 300 nouns: -"ER"
                'ing',  # and -"ING"
                'ed',  # and two adjectives, -"ING" and -"ED".
                'ly',  # Adverbs use -"LY" from qualifiers.
                'est',  # Be prepared to find -"ER" and -"EST" in common usage.
                'able',
                'ful',
                'keeper',
                'like',
                'man',
                'men',
                'woman',
                'women',
                'way',
                'work',
                'works',
                'worker'
                ]

    vocab = {}
    try:
        for a in model.vocab.keys():
            vocab[a] = a
        print 'First one'
    except:
        for a in model.wv.vocab.keys():
            vocab[a] = a
        print 'second'
    try:
        d = pickle.load(open('../data/basic_english.pickle', 'rb'))
    except:
        d = {}
    st = LancasterStemmer()
    for i, word in enumerate(lst):
        if i % 5 == 0:
            print 'Make Dictionary {} of {}'.format(i, len(lst))
        b = pos_tag([word])[0][1]
        d[word] = [word, word, b]
        for pre in prefixes:
            for suf in suffixes:
                chk = pre + word + suf
                if chk in vocab and len(chk) > 2:
                    b = pos_tag([chk])[0][1]
                    d[chk] = [chk, word, b]
                    # print chk, word, b
                # check double consonents
                if word[len(word) - 1] not in 'aeiou':
                    chk = pre + word + word[len(word) - 1] + suf
                    if chk in vocab and len(chk) > 2:
                        b = pos_tag([chk])[0][1]
                        d[chk] = [chk, word, b]
                        # print chk, word, b
                # drop last
                if word[len(word) - 1] in 'fye' and len(word) > 2:
                    chk = pre + word[:-1] + suf
                    if chk in vocab and len(chk) > 2:
                        b = pos_tag([chk])[0][1]
                        d[chk] = [chk, word, b]
                        # print chk, word, b
                # wife -> wives...
                if word[len(word) - 2] == 'f' and len(word) > 3:
                    chk = pre + word[:-2] + suf
                    if chk in vocab and len(chk) > 2:
                        b = pos_tag([chk])[0][1]
                        d[chk] = [chk, word, b]
                        # print chk, word, b
        with open('../data/basic_english.pickle', 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('../data/basic_english - Copy.pickle', 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print len(d)
    return d

if __name__ == '__main__':
    make_dict()
