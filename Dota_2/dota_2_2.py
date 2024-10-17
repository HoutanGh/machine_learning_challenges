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

print(data.head())

champion_columns = ['t1_champ1', 't1_champ2', 't1_champ3', 't1_champ4', 't1_champ5',
    't2_champ1', 't2_champ2', 't2_champ3', 't2_champ4', 't2_champ5'
]

# Get all unique champions
all_champions = pd.unique(data[champion_columns].values.ravel())
print(f'Total unique champions: {len(all_champions)}')

# using one-hot encoding

ohe = OneHotEncoder(handle_unknown='ignore')

ohe.fit(data[champion_columns])

champions_encoded = ohe.transform(data[champion_columns])

feature_names = ohe.get_feature_names_out(champion_columns)

encoded_data = pd.DataFrame(champions_encoded, columns=feature_names)

data_encoded = pd.concat([encoded_data, data['winner']], axis=1)

print(data_encoded)