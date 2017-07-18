# **Basic English Translator**
According to [http://ogden.basic-english.org/](http://ogden.basic-english.org/):
> If one were to take the 25,000 word Oxford Pocket English Dictionary and take away the redundancies of our rich language and eliminate the words that can be made by putting together simpler words, we find that 90% of the concepts in that dictionary can be achieved with 850 words. The shortened list makes simpler the effort to learn spelling and pronunciation irregularities. The rules of usage are identical to full English so that the practitioner communicates in perfectly good, but simple, English.

The goal of the Basic English Translator was to take this idea, specifically the [850 words](http://ogden.basic-english.org/words.html) proposed by **Charles K. Ogden** in the 1930s, each of the [international and supplementary words](http://ogden.basic-english.org/intlword.html) needed for various industries, and the [next steps words](http://ogden.basic-english.org/intlword.html) that English speakers should know, and simplify a given block of text to these words.

### **Research Question**
*Can we use Word-to-vector (gensim), parts of speech tagging (from nltk), and graph theory (networkx) to convert a block of text into Basic english?

## **Data Understanding**

* **Basic/Simple English:** There are versions of 'Simple' or 'basic' English, [Charles K Ogden](http://ogden.basic-english.org/)’s 850 words being the most common.  [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Simple_English_Wikipedia) being another (with a broader vocabulary).
* **GoogleNews-vectors-negative300.bin:** A set of words from Google News [already vectorized](https://github.com/mmihaltz/word2vec-GoogleNews-vectors/blob/master/GoogleNews-vectors-negative300.bin.gz), may help with translation of input text into simple English, at least until my model is ready.

## **Data Preparation**
Given how long *gensim* functions take to run, forcing a user to wait for *gensim* to calculate cosine similarities is not practical.  Instead, I will create a dictionary (in both the pythonic and literal sense of the term) to map English words to words that appear on Ogden's list.  The following flow chart shows this process:

![Flow Chart](/images/data_prep.png) Format: ![Blue -> BasicEnglishTranslator.py/graph_model.py Functions, Green -> gensim, Purple -> networkx, red -> nltk]

* **First, find "best" matches for a word:** This is done by taking a *gensim* model (in my case [google news](https://github.com/mmihaltz/word2vec-GoogleNews-vectors/blob/master/GoogleNews-vectors-negative300.bin.gz))...
  * Take each word, and find the top 10 connections (by cosine similarity)
  * Keep only the top 3 (that match in *parts of speech*)
  * Remove the worst match (using *gensim*)
  * Store in a dictionary
* **Next, make a weighted graph** where each word connects to the two words it was matched to, plus any words matched to it.  Set the weights of the graph to "(Sine Similarity)<sup>2</sup>," i.e.  1 - (cosine similarity)<sup>2</sup>.
* **Find shortest path to Basic English words**, starting from the most complex and heading down to least (so that the model favors more simplistic words).
* **Make a dictionary** that stores this information, save in a cPickle file so we can access it whenever we need to translate text.

## **Evaluation**
1. **Checking Basic English Translator:** For this I scraped some articles from both [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page) and [Standard English Wikipedia](https://en.wikipedia.org/wiki/Main_Page).  I then calculated the complexity of the text, using [Flesch-Kincaid reading levels](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests), and compared the original documents to the translations provided by my model.
    * ![Flow Chart](/images/flesch-kincaid_graph.png) Format: ![Flesch-Kincaid scores of original document on the x-axis, and the difference between the translated and the original on the y.]

2. **Look at Actual Text** While not effective to check everything, it does allow for some basic intuition on the text itself.  Example below is the simple Wikipedia article for ["spoon"](https://simple.wikipedia.org/wiki/Spoon):
    * **Early Model:** *another spoon is another instrument for eating. it is sometimes used for eating foods that are like liquids (like soup and soy), and it might also be used for stirring. humans use spoons every day. spoons are mostly useful for eating liquids, such as soup, though some solids (like tapioca and ice butter) are also sometimes eaten with spoons. another ladle is another kind of serving spoon used for soup, lager, or other foods. there are many different kinds of spoons. there are dessert spoons, soup spoons, baby spoons, teaspoons, thirds and others. there are also spoons that are collector whips and are monopolist another farmer of money. some performers even use two spoons as another musical instrument like another castanet. spoons have been used as computers for eating since paleolithic times. prehistoric peoples probably used shells, or small sheets of wood as spoons. both the testament and latin words for spoon come from the word superposition, which is another spiral-shaped lager shell. the anglo-saxon word spoon, means another sidewalk or spicy of wood.*
  * **Current Model:** *INSERT NEW MODEL STUFF HERE*
3. **Look at word associations in a graph** keeping the begining of each path in the final dictionary, I can examine *any* connection made by looking at a simple graph for each word.  The graph highlights the word with a yellow dot for ease of searching, and has three colors: blue words are words from [Ogden's Basic English](http://ogden.basic-english.org/), red words are words that are variations of the basic words, and represent stopping points in the translation, and green words are words that will be replaced...just keep tracing out the path until a red or blue word is hit.
    * Insert a graph
    * Insert a graph
    * Insert a graph

## **Deployment**
To use the program, go to [www.BasicEnglishTranslator.com](http://www.basicenglishtranslator.com/).
