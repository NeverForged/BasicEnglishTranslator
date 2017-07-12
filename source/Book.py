import os
import sys
import pandas as pd
import unicodedata
import numpy as np
from nltk.corpus import stopwords
from nltk.help import upenn_tagset
from sklearn.feature_extraction.text import CountVectorizer
import string

class Book:
    '''
    Class handler for books from project Gutenberg
    '''
    def __init__(self, book):
        self.title = ''
        self.author = ''
        n = len(book)
        lim = 30
        if lim > n:
            lim = n
        # Project Gutenburg has no way to determine front matter,
        # so cut first 5% to avoid it
        start = int(n - 0.95*n)
        for line in book[:lim]:
            if 'Title: ' in line:
                self.title = line.strip().replace('Title: ','')
                # print self.title
            if 'Author: ' in line:
                self.author = line.strip().replace('Author: ','')

        # start set to strip off various lengths of front-matter

        temp_text = "".join(book[start:]).encode('ascii', 'replace')
        # -19350 to strip off the Project Gutenberg donation requests
        self.text = temp_text[0:len(temp_text) - 19350]
        self.clean_text = ''
        self.meaningful_words = []

    def clean(self):
        # Get rid of line breaks.
        x = self.text.split("\n")
        x = [i.strip() for i in x]
        self.clean_text = " ".join(x)

        # Lower case.
        self.clean_text = self.clean_text.lower()

        # Remove punctuations
        prt = set(string.printable)
        self.clean_text = filter(lambda x: x in prt, self.clean_text)
        self.clean_text = self.clean_text.strip(string.punctuation)
        # Remove spaces.
        self.clean_text = self.clean_text.replace("  "," ")
        while "  " in self.clean_text:
            self.clean_text = self.clean_text.replace("  "," ")


    def remove_stop_words(self):
        stops = set(stopwords.words("english"))
        words = self.clean_text.split()
        self.meaningful_words = [w for w in words if not w in stops]

    def __getitem__(self, index):
        if index == 0:
            return self.clean_text
        if index == 1:
            return self.title
        if index == 2:
            return self.author
