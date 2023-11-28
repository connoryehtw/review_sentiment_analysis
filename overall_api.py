def overall_sa(toAna, stop_words, lemmatizer, word_tokenize, sent_tokenize):
    import pandas as pd
    import pickle

    # load necessary package from the pickle file
    with open('overall_model.pkl', 'rb') as file:
        model, tfidf, original_columns = pickle.load(file)
    
    # get integer rating and remove reviews that have no comment 
    toAna['rating'] =  toAna['rating'].str.split(" ").str[0].astype(int)
    toAna = toAna.dropna(subset=['review'])

    # tokenize and lemmatize reviews and store to review_processed column
    def preprocess_review(review):
        sentences = sent_tokenize(review)
        words = [word_tokenize(sentence) for sentence in sentences]
        words = [[word.lower() for word in sentence if word.lower() not in stop_words] for sentence in words]
        words = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in words]
        return ' '.join([' '.join(sentence) for sentence in words])

    toAna['review_processed'] = toAna['review'].apply(preprocess_review)
    toAna = toAna.reset_index()

    # transform the processed review into readale variables with tfidf for ML model
    text_tfidf = tfidf.transform(toAna["review_processed"])
    text_tfidf_dense = text_tfidf.toarray()
    feature_names = tfidf.get_feature_names_out()
    new_data_tfidf = pd.DataFrame(text_tfidf_dense, columns=feature_names)
    
    # add rating as one of a variable columns
    new_data_tfidf.insert(0, 'stars_review', toAna['rating'])
    
    # loop over features in the training set, fill features non exist in input data with 0, and reorder the features to match with training data
    def match_features(new_data, original_columns):
        for feature in original_columns:
            if feature not in new_data.columns:
                new_data[feature] = 0.0
        matched_data = new_data[original_columns]
        return matched_data
    matched_data_tfidf = match_features(new_data_tfidf, original_columns)
    
    # make predictions using the loaded model
    svm_predictions = model.predict(matched_data_tfidf)
    toAna['prediction'] = svm_predictions
    toAna = toAna.drop('index', axis=1)

    return toAna