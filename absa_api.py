def absa(new_data):
    import pickle
    from scipy.sparse import hstack

    # load necessary package from the pickle file
    with open('absa_model.pkl', 'rb') as file:
        loaded_dict = pickle.load(file)

    loaded_model = loaded_dict['model']
    loaded_vectorizer_aspect = loaded_dict['vectorizer_aspect']
    loaded_vectorizer_adj = loaded_dict['vectorizer_adj']
    loaded_vectorizer_overall_predictions = loaded_dict['vectorizer_overall_predictions']
    loaded_label_encoder = loaded_dict['label_encoder']

    # transform the variables into readale form for ABSA model
    new_aspect = loaded_vectorizer_aspect.transform(new_data['aspect'])
    new_adj = loaded_vectorizer_adj.transform(new_data['adjective'])
    new_overall_predictions = loaded_vectorizer_overall_predictions.transform(new_data['predictions_overall'])

    new_data_vec = hstack([new_aspect, new_adj, new_overall_predictions])

    # make predictions
    predictions = loaded_model.predict(new_data_vec)

    # decode the predictions
    decoded_predictions = loaded_label_encoder.inverse_transform(predictions)

    new_data['predictions_absa'] = decoded_predictions
    return new_data