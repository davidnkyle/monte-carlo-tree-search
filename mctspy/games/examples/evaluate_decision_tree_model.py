
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import glob
import time
import psutil
import os

startTime = time.time()

folder_path = r'G:\Users\DavidK\analyses_not_project_specific\20220831_simulation_results\20220907_results/*.csv'
all_files = glob.glob(folder_path)
li = []

print('reading data')

for filename in all_files:
    df = pd.read_csv(filename, index_col='Unnamed: 0')
    li.append(df)
    print('.', end='')


df = pd.concat(li)
del li

# print()
# executionTime = (time.time() - startTime) / 60
# print('Execution time in minutes: ' + str(executionTime))
# print('total: {} MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
#
# split = int((len(df.index)*2)//3)
# df_training = df.iloc[0:split]
# df_validation = df.iloc[split:]
#
# y_train = np.array(df_training['result'])
# x_train = df_training.drop('result', axis=1).values
#
# print('fitting model')
# model = DecisionTreeClassifier(criterion='entropy')
# model.fit(x_train, y_train)
#
# y_validate = np.array(df_validation['result'])
# x_validate = df_validation.drop('result', axis=1).values
#
# print('predicting results')
# y_pred = model.predict(x_validate)
#
# print(np.absolute(y_pred-y_validate).sum()/len(df_validation.index))
# print(y_validate.mean())
# print(y_pred.mean())
# print('false positives {}'.format(1 - y_pred[np.where(y_validate==1)].mean()))
# print('false negatives {}'.format(y_pred[np.where(y_validate==0)].mean()))

# import pickle
# with open('model_3pl_round12.pkl','wb') as f:
#     pickle.dump(model,f)

executionTime = (time.time() - startTime) / 60
print('Execution time in minutes: ' + str(executionTime))
print('total: {} MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

