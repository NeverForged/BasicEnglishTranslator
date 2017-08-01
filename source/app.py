import sys
import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Appearance import Appearance
from BasicEnglishTranslator import BasicEnglishTranslator
from flask import Flask, render_template, request, jsonify, redirect

app = Flask(__name__)
@app.route('/process', methods=['POST'])

@app.route('/draw/<specs>.png')
def draw(specs):
    '''
    Not part of the basic english translator, but part of a different project
    placing here since I don't want to create another ec2 instance for that
    project yet.
    '''
        #if not os.path.exists('/static/images/' + self.specs + '.png'):
    print('need to draw this combination')
    fig, ax = plt.subplots(1, figsize=(4.0, 6.0))
    appr = Appearance(None, None, ax, specs)
    appr.draw_char()
    appr.show()
    # return redirect("../source/static/images/" + specs + ".png", code=302)
    return redirect('/static/images/temp.png')

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
