# import all the necessary libraries
from flask import Flask, request, jsonify
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
# from spacy import displacy

# import all the dependent modules
import overall_api
import aspect_ext_api
import absa_api


# download necessary NLTK corpora
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# load necessary objects
stop_words = set(stopwords.words('english'))
stop_words.update(['way', 's', 'pre', 'flight', '-', 'next', 'one', 'different', 'nothing', 'one', 'you', 'i'])
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

# initialize the flask app
app = Flask(__name__)

# python decorator that triggers function below with specified URL
@app.route('/analyze-review', methods=['POST'])
def analyze_review():
    # extract the review and its rating from the request
    data = request.get_json()
    review_text = data.get('review_text')
    rating = data.get('rating')
    
    # convert the review into a pandas DataFrame
    review = pd.DataFrame([(review_text, rating)], columns=['review', 'rating'])
    # perform sentiment analysis
    try:
        overall_prediction = overall_api.overall_sa(review, stop_words, lemmatizer, word_tokenize, sent_tokenize)
        overall_sentiment = overall_prediction.iloc[0]['prediction']
    except:
         overall_sentiment = "Neutral"
    # perform aspect extraction, adjective pairing, and aspect-based sentiment analysis
    try:
        extracted = aspect_ext_api.aspects_extraction(overall_prediction, stop_words, lemmatizer, nlp)
        absa_predicted = absa_api.absa(extracted)
    except:
        absa_predicted = pd.DataFrame()

    # prepare the output
    result = {
        'overall_sentiment_prediction': overall_sentiment,
        'absa': []
    }
    
    for _, row in absa_predicted.iterrows():
        result['absa'].append({
            'aspect': row['aspect'],
            'adjective': row['adjective'],
            'absa_prediction': row['predictions_absa']
        })
    
    # return the result
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)