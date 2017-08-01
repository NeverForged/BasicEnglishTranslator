import sys
import cPickle as pickle
from BasicEnglishTranslator import BasicEnglishTranslator
from flask import Flask, render_template, request, jsonify, redirect

app = Flask(__name__)
@app.route('/process', methods=['POST'])

@app.route('/draw/<specs>.png')
def draw(specs):
    n = len(specs)
    f = '/static/images/neverforged_logo.png'
    if n % 2 == 0:
        f = '/static/images/neverforged_logo_w.png'
    return redirect(f)

# load page when someone hits the site
@app.route('/', methods=['GET'])
def index():
    return render_template('submit.html')


# button functionality
@app.route('/process', methods=['POST'])
def process():
    user_data = request.json
    print user_data
    X = user_data['text_input']
    # print X
    y = 'fail'
    try:
        print 'before...'
        y = '{}'.format(translate_text(X))
        print y
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
        y = str(e).replace("<", "").replace(">", "")
    return jsonify({'Input': X, 'Classification': y})


def translate_text(text_input):
    '''
    Calls BasicEnglishTranslator and fits the text.
    '''
    d_name = '../data/basic_english.pickle'
    dictionary = pickle.load(open(d_name, "rb"))
    trans = BasicEnglishTranslator(basic_dictionary=dictionary, verbose=True)
    return trans.fit(text_input)

if __name__ == '__main__':
    # run it...
    app.run(host='0.0.0.0', port=8880, debug=True)
