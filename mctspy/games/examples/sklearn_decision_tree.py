
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pickle


df = pd.read_csv('pl3_round13_1000000_20220821.csv', index_col='Unnamed: 0')


y_train = np.array(df['result'])
x_train = df.drop('result', axis=1).values

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)


with open('model_3pl_round13.pkl','wb') as f:
    pickle.dump(model,f)

