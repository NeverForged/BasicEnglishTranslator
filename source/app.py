from flask import Flask, render_template, request, jsonify
import cPickle as pickle
from BasicEnglishTranslator import BasicEnglishTranslator
import sys
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('submit.html')

@app.route('/process', methods=['POST'])
def process():
    user_data = request.json
    print user_data
    X = user_data['text_input']
    # print X
    y = 'fail'
    try:
        print 'before...'
        y = '{}'.format(classify_text(X))
        print y
    except: # catch *all* exceptions
          e = sys.exc_info()[0]
          y = str(e).replace("<","").replace(">","")
    return jsonify({'Input': X, 'Classification':y})


def classify_text(text_input):
    '''
    '''
    trans = BasicEnglishTranslator()
    return trans.fit(text_input)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8880, debug=True)
