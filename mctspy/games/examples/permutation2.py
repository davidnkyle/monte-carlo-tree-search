from copy import copy
import random

import pandas as pd

possibility_matrix = pd.read_csv('possibility_matrix .csv')
starting_matrix = pd.read_csv('starting_matrix.csv', index_col='Unnamed: 0')

starting_deck = [starting_matrix.loc[starting_matrix.iloc[:, i]==1].index[0] for i in range(len(starting_matrix.columns))]


relative_frequencies = pd.read_csv('relative_frequency.csv', index_col='Unnamed: 0')
relative_frequencies.columns = range(len(starting_deck))
totals = pd.DataFrame(data=0, index=relative_frequencies.index, columns=range(len(starting_deck)))



def swap(starting_deck, rf, N):
    # todo implement hexagon swap for non-monotonic matrices
    A = copy(starting_deck)
    for _ in range(N):
        i, j = random.choices(range(len(A)), k=2)
        curr = rf.at[A[i], i]*rf.at[A[j], j]
        new = rf.at[A[i], j]*rf.at[A[j], i]
        p = new/(curr + new)
        if random.random() < p:
            A[i], A[j] = A[j], A[i]
    return A

new = starting_deck
for _ in range(1000):
    new = swap(new, relative_frequencies, 20)
    for idx in range(len(new)):
        totals.loc[new[idx], idx] += 1
