from flask import Flask, request, render_template
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords
import pickle
import pandas as pd
import string
import waitress



app = Flask(__name__)

model = pickle.load(open("XGBmodel.pkl", "rb"))
bow = pickle.load(open("count_vectorizer.pkl", "rb"))

# function lowers the characters

def remove_numericals(a_string):
    
    table = str.maketrans('', '', string.digits)
    new_string = a_string.translate(table)
    
    return new_string

#remove html tags
def clean_html(text):
    clean=re.compile('<.*?>')
    return re.sub(clean,'',text)


def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)


def remove_punctuations(a_string):
    
    new_string = a_string.translate(str.maketrans('', '', string.punctuation))
    
    
    return new_string


@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    
    #convert to lowercase
    text1 = request.form['text1'].lower()
    text1 = remove_numericals(text1)
    text1 = clean_html(text1)
    text1 = remove_emojis(text1)
    processed_doc1 = remove_punctuations(text1)
    test_case = bow.transform([processed_doc1])
    test_case = pd.DataFrame(test_case.todense())
    prediction = model.predict(test_case)[0]
    
    Ratings = {0 : 'Negative',
               1 :  'Neutral',
               2 :  'Positive'}

    return render_template('form.html', final=Ratings[prediction], text1=text1)

if __name__ == "__main__":
    from waitress import serve
    app.run(debug=True, host="0.0.0.0", port=8080, threaded=True)
