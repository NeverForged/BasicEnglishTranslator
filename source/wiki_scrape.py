import time
import random
import requests
import nltk.data
import pandas as pd
import cPickle as pickle
from bs4 import BeautifulSoup


def main():
    '''
    A simple script to scrape wikipedia for articles of the same name as words
    in Ogden's Basic English (Expanded).  This is to build out a model using
    intelligent connections to build a dictionary that won't be so news-focused
    '''
    basic = list(pd.read_csv('../data/basic_english_wordlist.csv')['WORDS'])
    basic = list(set(basic))
    basic.sort()
    print len(basic)
    sentences = []
    try:
        articles = pickle.load(open('../data/wikipedia.pickle', 'rb'))
    except:
        articles = {}
    for word in basic:
        start = time.clock()
        # avoid DOS against wikipedia
        if 'simple_' + word not in articles:
            try:
                item = '/wiki/' + word
                r = requests.get('https://simple.wikipedia.org'+item)
                soup = BeautifulSoup(r.content, 'html.parser')
                tags = soup.find_all('p')
                MyText = '\n'.join([tag.get_text() for tag in tags])
                tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                # Load in sentences
                MyText = MyText.encode('ascii', 'replace')
                try:
                    sentences = pickle.load(open('../data/sentences.pickle',
                                                 'rb'))
                except:
                    print 'No sentences saved.'
                sent = book_to_sentences(MyText, tokenizer)
                sentences += sent
                articles['simple_'+word] = sent
                with open('../data/sentences.pickle', 'wb') as handle:
                    pickle.dump(sentences, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
                with open('../data/wikipedia.pickle', 'wb') as handle:
                    pickle.dump(articles, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
                print '{} - {} ({:.2f}s)'.format('simple_'+word,
                                                 len(sentences),
                                                 time.clock() - start)
            except:
                print 'Failure....likely request'
        if 'en_' + word not in articles:
            # now add in en...
            try:
                item = '/wiki/' + word
                r = requests.get('https://en.wikipedia.org'+item)
                soup = BeautifulSoup(r.content, 'html.parser')
                tags = soup.find_all('p')
                try:
                    sentences = pickle.load(open('../data/sentences.pickle',
                                                 'rb'))
                except:
                    print 'No sentences saved.'
                MyText = '\n'.join([tag.get_text() for tag in tags])
                tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                MyText = MyText.encode('ascii', 'replace')
                sent = book_to_sentences(MyText, tokenizer)
                sentences += sent
                articles['en_'+word] = sent
                with open('../data/sentences.pickle', 'wb') as handle:
                    pickle.dump(sentences, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
                with open('../data/wikipedia.pickle', 'wb') as handle:
                    pickle.dump(articles, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
                print '{} - {} ({:.2f}s)'.format('en_'+word, len(sentences),
                                                 time.clock() - start)
            except:
                print 'Failure....likely request'


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
    main()
