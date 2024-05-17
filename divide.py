import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/variant_4.csv')
train, test = train_test_split(df, train_size=0.8, test_size=0.2, stratify=df['Scholarship holder'])

train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)