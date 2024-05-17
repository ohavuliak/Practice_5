import pandas as pd
import pickle
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from preprocessing import preprocess_train_data

def train_model(file_name, model_name):
    data = pd.read_csv('data/' + file_name)

    data = preprocess_train_data(data)

    X = data.drop(columns=['Scholarship holder'])
    Y = data['Scholarship holder']

    models = {
        'random_forest': RandomForestClassifier(verbose=0),
        'extra_trees': ExtraTreesClassifier(verbose=0),
        'lightgbm': LGBMClassifier(verbose=0),
        'catboost': CatBoostClassifier(verbose=0),
        'gradient_boosting': GradientBoostingClassifier(verbose=0)
    }

    model = models[model_name]
    model.fit(X, Y)

    with open(f'models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)

train_model('train.csv', 'gradient_boosting')