
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np


df = pd.read_csv('pl3_round13_1000000_20220821.csv', index_col='Unnamed: 0')


split = int(len(df.index)//2)
df_training = df.iloc[0:split]
df_validation = df.iloc[split:]

y_train = np.array(df_training['result'])
x_train = df_training.drop('result', axis=1).values

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)

y_validate = np.array(df_validation['result'])
x_validate = df_validation.drop('result', axis=1).values

y_pred = model.predict(x_validate)

print(np.absolute(y_pred-y_validate).sum()/len(df_validation.index))
print(y_validate.mean())
print(y_pred.mean())
print('false positives {}'.format(1 - y_pred[np.where(y_validate==1)].mean()))
print('false negatives {}'.format(y_pred[np.where(y_validate==0)].mean()))

