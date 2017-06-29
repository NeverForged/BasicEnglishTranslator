# **Ghost Writer**
The author *Alexandre Dumas (pere)* was known for his writing collaborations; other authors (notably *August Maquet*) would write the plot and outline, Dumas would fill in the details, add dialogue, etc.

With the large number of scanned books by famous authors one can find at *Project Gutenberg* [(gutenberg.org)](https://www.gutenberg.org/), one could easily have a great author such as *Dumas* do the same for their own outlines, thanks to natural language processing.  Let famous authors fill in the details of, or lend their style to:

* social media posts
* plot outlines
* blog entries

...you name it.

### **Research Question**
*Can we use NLP and other tools to turn basic plot outlines into stories from famous authors?*
* *Sub Question:* Can we use gensim/nlp to convert a block of text into Basic english?
* *Sub Question:* Can I use nlp, n-grams, word2vec, etc. to rewrite a simple english block of text in the style of a famous, dead author?

## **Data Understanding**

* **Project Gutenberg:** A [wonderful site](https://www.gutenberg.org/) that provides a large corpus of texts from various authors.  They are in utf8, and have various degrees of front-matter, and 19,300 characters of added information on the back end, but they are easy enough to grab, vectorize, etc.
* **Basic/Simple English:** There are versions of 'Simple' or 'basic' English, [Charles K Ogden](http://ogden.basic-english.org/)’s 850 words being the most common.  [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Simple_English_Wikipedia) being another (with a broader vocabulary).
* **GoogleNews-vectors-negative300.bin:** A set of words from Google News [already vectorized](https://github.com/mmihaltz/word2vec-GoogleNews-vectors/blob/master/GoogleNews-vectors-negative300.bin.gz), may help with translation of input text into simple English, at least until my model is ready.

## **Data Preparation**
1. **Get text to 'ghost write'**  This could be web scrapped from somewhere, a user input into a text-box on a website, etc.
2. **Convert text to "_Basic English_"**  The python file [BasicEnglishTranslator.py](https://github.com/NeverForged/GhostWriter/blob/master/source/BasicEnglishTranslator.py) takes an input text and does the following:
    * *Breaks it up into a list of words and their parts of speech* using <nltk.pos_tag> which tags each word and symbol with a part of speech.
    * *Checks part of speech to see if the word* should *be converted.*  'Should' is an operative term here; currently the file avoids proper nouns, punctuation, and certain adjectives.  The adjective type *JJ* or *adjective or numeral, ordinal* was added to pass-thru list due to the tendency for my model to convert numbers to the wrong numbers, 'thousands' to 'hundreds' for example.
    * *Checks if the word is known to it's Basic English dictionary.*  If so, it finds the Basic English equivalent and replaces it.  if not, it uses a <gensim wor2vec> model (such as [Google News](https://github.com/mmihaltz/word2vec-GoogleNews-vectors/blob/master/GoogleNews-vectors-negative300.bin.gz)) to find equivalent words and check if those are known to it; if so it updates a version of the model, if not, it leaves the word alone.
    
