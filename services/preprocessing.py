import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.simplefilter('ignore')

def preprocess_train_data(data: pd.DataFrame) -> pd.DataFrame:
    # dropping unnecessary columns
    data = data.drop(columns=['Unnamed: 0'])

    # handling missing values using mode
    missing_cols = sorted(data.columns[data.isnull().any()].tolist(), key=lambda col: data[col].isnull().sum(), reverse=True)
    modes = {col: data[col].mode()[0] for col in data.columns}
    for col in missing_cols:
        data[col] = data[col].fillna(modes[col])
    
    #saving modes in a file
    with open('services/settings/modes.pkl', 'wb') as f:
        pickle.dump(modes, f)
    
    # encoding categorical columns
    categorical_cols = data.select_dtypes(include='object').columns

    onehot_cols = [col for col in categorical_cols if len(data[col].unique()) > 2]
    label_cols = [col for col in categorical_cols if len(data[col].unique()) == 2]

    onehot_encoder = OneHotEncoder(handle_unknown='ignore')

    onehot_encoded = pd.DataFrame(onehot_encoder.fit_transform(data[onehot_cols]).toarray())
    onehot_encoded.columns = onehot_encoder.get_feature_names_out(onehot_cols)
    
    label_encoders = {col: LabelEncoder() for col in label_cols}
    label_encoded = data[label_cols].apply(lambda col: label_encoders[col.name].fit_transform(col))

    # saving encoders in a file
    with open('services/settings/onehot_encoder.pkl', 'wb') as f:
        pickle.dump(onehot_encoder, f)
    with open('services/settings/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # replacing categorical columns with encoded columns
    data = data.drop(columns=categorical_cols)
    data = pd.concat([data, onehot_encoded], axis=1)
    data = pd.concat([data, label_encoded], axis=1)

    return data

def preprocess_testing_data(data: pd.DataFrame) -> pd.DataFrame:
    # dropping unnecessary columns
    data = data.drop(columns=['Unnamed: 0'])

    # handling missing values using mode
    missing_cols = sorted(data.columns[data.isnull().any()].tolist(), key=lambda col: data[col].isnull().sum(), reverse=True)
    with open('services/settings/modes.pkl', 'rb') as f:
        modes = pickle.load(f)
    
    for col in missing_cols:
        data[col] = data[col].fillna(modes[col])
    
    # encoding categorical columns
    with open('services/settings/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('services/settings/onehot_encoder.pkl', 'rb') as f:
        onehot_encoder = pickle.load(f)

    categorical_cols = data.select_dtypes(include='object').columns

    onehot_cols = [col for col in categorical_cols if len(data[col].unique()) > 2]
    label_cols = [col for col in categorical_cols if len(data[col].unique()) == 2]

    onehot_encoded = pd.DataFrame(onehot_encoder.transform(data[onehot_cols]).toarray())
    onehot_encoded.columns = onehot_encoder.get_feature_names_out(onehot_cols)
    
    label_encoded = data[label_cols].apply(lambda col: label_encoders[col.name].transform(col))
    
    # replacing categorical columns with encoded columns
    data = data.drop(columns=categorical_cols)
    data = pd.concat([data, onehot_encoded], axis=1)
    data = pd.concat([data, label_encoded], axis=1)

    return data

