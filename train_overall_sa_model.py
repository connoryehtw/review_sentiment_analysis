# import all the necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler

# download necessary NLTK corpora
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_review(review):
    # tokenize reivew
    sentences = sent_tokenize(review)
    words = [word_tokenize(sentence) for sentence in sentences]
    # remove stop words in reviews
    words = [[word.lower() for word in sentence if word.lower() not in stop_words] for sentence in words]
    # lemmatize the words
    words = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in words]
    # return a single string
    return ' '.join([' '.join(sentence) for sentence in words])

# load crawled reviews
review = pd.read_csv("review.csv")
# clean and preprocess review
review['rating'] =  review['rating'].str.split(" ").str[0].astype(int)
review = review.dropna(subset=['review'])
review['review_processed'] = review['review'].apply(preprocess_review)

# load labeled data
labeled_df1 = pd.read_csv("labeled_data/labeled1.csv")
labeled_df2 = pd.read_csv("labeled_data/labeled2.csv")

# concatenate and preprocess labeled data
labeled_df = pd.concat([labeled_df1, labeled_df2], ignore_index=True)
labeled_df = labeled_df.dropna(subset=['Review'])
labeled_df['review_processed'] = labeled_df['Review'].apply(preprocess_review)

# create TF-IDF features to consider frequency
tfidf = TfidfVectorizer()
tfidf.fit(labeled_df["review_processed"])
text_tfidf = tfidf.transform(labeled_df["review_processed"])
text_tfidf_dense = text_tfidf.toarray()
feature_names = tfidf.vocabulary_.keys()
df_tfidf = pd.DataFrame(text_tfidf_dense, columns=feature_names)
df_tfidf.insert(0, 'stars_review', labeled_df['Star'])

# split the dataset into training and testing sets
X = df_tfidf
y = labeled_df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=78)

# tackle unbalanced data with oversampling
ros = RandomOverSampler(random_state=44)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# train and evaluate models
models = [
    ('SVM', SVC(class_weight='balanced'), {'C': [0.1, 0.5, 1, 10], 'kernel': ['linear', 'rbf']})
    , ('NB', MultinomialNB(), {'alpha': [0.1, 0.5, 1, 10]})
    , ('RF', RandomForestClassifier(class_weight='balanced'), {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]})
    , ('LR', LogisticRegression(class_weight='balanced'), {'C': [0.1, 0.5, 1, 10], 'penalty': ['l1', 'l2', 'None', 'elasticnet']})
]

# identify the best model
for name, model, params in models:
    grid = GridSearchCV(model, params, cv=5)
    grid.fit(X_train_resampled, y_train_resampled)
    best_params = grid.best_params_
    best_model = grid.best_estimator_
    predictions = best_model.predict(X_test)
    report = classification_report(y_test, predictions)
    print(f"{name}:")
    print(report)

#--------------after evaluation---------------
# the best model is LogisticRegression (accuracy rate 0.93)
# dump the best model into the pickle file
import pickle
with open('overall_model.pkl', 'wb') as file:
    pickle.dump((best_model, tfidf, df_tfidf.columns), file)