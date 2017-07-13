import make_dictionary as md
import gensim
import build_model as mm
from BasicEnglishTranslator import BasicEnglishTranslator
import cPickle as pickle
import time

def main():
    '''
    A simple 'reset' script for the dictionary; allows a rebuild when enough
    new sentences have been added to data/sentences.pickle
    '''
    d = {}
    with open('../data/basic_english.pickle', 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../data/basic_english - Copy.pickle', 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print "making model..."
    start = time.clock()
    model = mm.main()
    # b = '../model/GoogleNews-vectors-negative300.bin'
    # try:
    #     model = gensim.models.KeyedVectors.load_word2vec_format(b, binary=True)
    # except:
    #     model = gensim.models.Word2Vec.load(b)
    s2 = time.clock()
    print '   ...took {:.3f}s \n'.format(s2-start)
    print 'making dictionary'
    s3 = time.clock()
    d = md.make_dict()
    print '   ...took {:.3f}s - {}\n'.format(s3-s2, len(d))
    print "Starting work..."
    sentences = pickle.load(open('../data/sentences.pickle', 'rb'))
    length = len(sentences)
    start =  time.clock()
    for i, sentence in enumerate(sentences):
        text = ' '.join(sentence)
        trans = BasicEnglishTranslator(model,
                                       basic_dictionary=d,
                                       threshold=1,
                                       verbose=False)
        trans.fit(text)
        d = trans.save_dictionary
        if i % 50 == 0:
            print "Sentences {} of {}: {} {:.2f}s".format(i, length, len(d),
                                                time.clock() - start)
            start = time.clock()
    with open('../data/basic_english.pickle', 'wb') as handle:
        pickle.dump(d,handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
