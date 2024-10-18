import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = pd.read_csv('trainingdata.csv', header=None, index_col=False)

# print(data.head())
# data = data[1:]
# print(data.head())
data.columns = ['index', 't1_champ1', 't1_champ2', 't1_champ3', 't1_champ4', 't1_champ5',
    't2_champ1', 't2_champ2', 't2_champ3', 't2_champ4', 't2_champ5',
    'winner']
data = data.drop('index', axis=1)

print(data.head())

champion_columns = ['t1_champ1', 't1_champ2', 't1_champ3', 't1_champ4', 't1_champ5',
    't2_champ1', 't2_champ2', 't2_champ3', 't2_champ4', 't2_champ5'
]

# Get all unique champions
all_champions = pd.unique(data[champion_columns].values.ravel())
print(f'Total unique champions: {len(all_champions)}')

# using one-hot encoding

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

champions_encoded = ohe.fit_transform(data[champion_columns])

feature_names = ohe.get_feature_names_out(champion_columns)

encoded_data = pd.DataFrame(champions_encoded, columns=feature_names)

data_encoded = pd.concat([encoded_data, data['winner'].reset_index(drop=True)], axis=1)

print(data_encoded.head())

data_cleaned = data_encoded.drop(index=0)
# data_cleaned.to_csv('data_cleaned.csv')

data_cleaned['winner'] = data_cleaned['winner'].map({1: 0, 2: 1})

print(data_cleaned['winner'].value_counts())

X = data_cleaned.drop('winner', axis=1)
y = data_cleaned['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# exact same accuracy, IDK why I thought this different way to one-hot encode was gonna be better

importances = model.feature_importances_

feature_importances_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
})

feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)

print(feature_importances_df.head(10))

# just to see whats affecting the model, IDK how one guy got 100%
