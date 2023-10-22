# import all the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack

# load labeled data
df = pd.read_csv('labeled_data/extracted_label_data.csv')
X = df[['aspect', 'adjective', 'predictions_overall']]
y = df['predictions_absa_pre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=34)

# create TF-IDF features to consider frequency
vectorizer_aspect = TfidfVectorizer()
X_train_aspect = vectorizer_aspect.fit_transform(X_train['aspect'])
X_test_aspect = vectorizer_aspect.transform(X_test['aspect'])

vectorizer_adj = TfidfVectorizer()
X_train_adj = vectorizer_adj.fit_transform(X_train['adjective'])
X_test_adj = vectorizer_adj.transform(X_test['adjective'])

vectorizer_overall_predictions = TfidfVectorizer()
X_train_overall_predictions = vectorizer_overall_predictions.fit_transform(X_train['predictions_overall'])
X_test_overall_predictions = vectorizer_overall_predictions.transform(X_test['predictions_overall'])

# stack matrices horizontally to a single matrix 
X_train_vec = hstack([X_train_aspect, X_train_adj, X_train_overall_predictions])
X_test_vec = hstack([X_test_aspect, X_test_adj, X_test_overall_predictions])

# encode y into numbers
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# perform SMOTE for oversampling
smote = SMOTE(random_state=43)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train_enc)

models = [
    ('NB', MultinomialNB(), {'alpha': [0.1, 0.5, 1, 10]})
    , ('RF', RandomForestClassifier(), {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30]})
    , ('LR', LogisticRegression(), {'C': [0.1, 0.5, 1, 10], 'penalty': ['l1', 'l2', 'None', 'elasticnet']})
    , ('SVM', SVC(), {'C': [0.1, 0.5, 1, 10], 'kernel': ['linear', 'rbf']})
]

for name, model, params in models:
    print(f"training: {name}...")
    grid = GridSearchCV(model, params, cv=5)
    grid.fit(X_train_resampled, y_train_resampled)
    best_params = grid.best_params_
    best_model = grid.best_estimator_
    predictions = best_model.predict(X_test_vec)
    report = classification_report(y_test_enc, predictions)
    print(f"{name}:")
    print(report)
    print("Best Parameters:")
    print(best_params)
    print("-------------------------")
    
#--------------after evaluation---------------
# the best model is SVM (accuracy rate 0.92)
# dump the best model into the pickle file
best_svm_model = grid.best_estimator_
import pickle
models_and_transformers = {
    'model': best_svm_model,
    'vectorizer_aspect': vectorizer_aspect,
    'vectorizer_adj': vectorizer_adj,
    'vectorizer_overall_predictions': vectorizer_overall_predictions,
    'label_encoder': label_encoder
}

with open('absa_model.pkl', 'wb') as file:
    pickle.dump(models_and_transformers, file)