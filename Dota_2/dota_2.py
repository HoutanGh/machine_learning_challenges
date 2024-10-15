import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

trainingdata = pd.read_csv('trainingdata.txt', header=None)
# print(trainingdata)

# trainingdata.to_csv('traingingdata.csv')

cleaned_data = trainingdata.reset_index(drop=True)
# print(cleaned_data.head())

# if the hero is in the game or not their column gets a 1 or 0 respectively
one_hot_data = pd.get_dummies(cleaned_data.iloc[:, :-1])

# print(one_hot_data.head())

labels = trainingdata.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(one_hot_data, labels, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_rep}')



