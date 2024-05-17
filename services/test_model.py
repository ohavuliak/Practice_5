import pandas as pd
import pickle
from preprocessing import preprocess_testing_data

def test_model(file_name, model_name):
    data = pd.read_csv('../data/' + file_name)

    data = preprocess_testing_data(data)

    X = data.drop(columns=['Scholarship holder'])
    Y = data['Target']

    with open('../services/settings/modes.pkl', 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X)

    pd.DataFrame(predictions).to_csv('../data/predictions.csv', index=False)

    accuracy = (predictions == Y).mean()
    print(f'Accuracy: {accuracy}')

test_model('test.csv', 'gradient_boosting')
