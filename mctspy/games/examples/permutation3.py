from copy import copy
import random

import pandas as pd

possibility_matrix = pd.read_csv('possibility_matrix2.csv', index_col='Unnamed: 0')
possibility_matrix.columns = range(3)
features_example = pd.read_csv('features_example.csv', index_col='Unnamed: 0')
features_example.columns = range(3)
starting_matrix = pd.read_csv('starting_matrix2.csv', index_col='Unnamed: 0')
starting_cards = pd.Series(index=starting_matrix.index)
for idx, row in starting_matrix.iterrows():
    starting_cards[idx] = int(row.loc[row==1].index[0])

# starting_deck = [starting_matrix.loc[starting_matrix.iloc[:, i]==1].index[0] for i in range(len(starting_matrix.columns))]


relative_frequencies = pd.read_csv('relative_frequencies2.csv', index_col='Unnamed: 0')
relative_frequencies.columns = range(3)
totals = pd.DataFrame(data=0, index=relative_frequencies.index, columns=range(3))



def swap(starting_deck, rf, N):
    # todo implement hexagon swap for non-monotonic matrices
    A = copy(starting_deck)
    for _ in range(N):
        i, j = random.choices(A.index, k=2)
        curr = rf.at[i, A[i]]*rf.at[j, A[j]]
        new = rf.at[i, A[j]]*rf.at[j, A[i]]
        p = new/(curr + new)
        if random.random() < p:
            A[i], A[j] = A[j], A[i]
    return A

def relative_frequency_from_features(features, possible):
    cards = possible.index
    players = possible.columns
    onlies = features.loc[['only_{}'.format(c) for c in cards], players]
    multis = features.loc[['has_{}'.format(c) for c in cards], players]
    rf = pd.DataFrame(index=cards, columns=players, data=onlies.values + multis.values)
    return rf


def hands_from_sample(sample, players):
    hands = [[] for _ in range(players)]
    for c, pl in sample.iteritems():
        hands[pl].append(c)
    return hands



new = starting_cards
for _ in range(1000):
    new = swap(new, relative_frequencies, 20)
    for idx in new.index:
        totals.loc[idx, new[idx]] += 1

cards = relative_frequencies.index
players = relative_frequencies.columns
onlies = features_example.loc[['only_{}'.format(c) for c in cards], players]
multis = features_example.loc[['has_{}'.format(c) for c in cards], players]
rf = pd.DataFrame(index=cards, columns=players, data=onlies.values + multis.values)


