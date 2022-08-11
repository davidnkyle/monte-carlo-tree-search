from copy import copy, deepcopy

import pandas as pd
import numpy as np
import random
import itertools
possibility_matrix = pd.DataFrame(data=np.ones((14, 3)), index=['b1', 'b2', 'b3', 'b4', 'g1', 'g2', 'g3', 'g4', 'p1', 'p2', 'p3', 'p4', 'z1', 'z2'])
cards_per_player = [5, 5, 4]
random.seed(987)

index = ['short_suited', 'only_1', 'only_2', 'only_3', 'only_4',
         'highest_1', 'highest_2', 'highest_3', 'highest_4',
         'lowest_1', 'lowest_2', 'lowest_3', 'lowest_4',
         'has_1', 'has_2', 'has_3', 'has_4',
         'no_1', 'no_2', 'no_3', 'no_4']

feature_probabilities_zero = pd.DataFrame({'b': [0.8, 0.1, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                           'g': [0, 0.1, 0.2, 0.1, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0, 0, 0.3, 0.5, 0.5, 0.3, 0.7, 0.5, 0.5, 0.7],
                                           'p': [0.4, 0, 0.1, 0.1, 0.1, 0, 0.1, 0.2, 0, 0.2, 0.1, 0, 0, 0.2, 0.4, 0.2, 0, 0.8, 0.6, 0.8, 1.0],
                                           'z': [0.7, 0.1, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]},
                                          index=index)

feature_probabilities_one = pd.DataFrame({'b': [0.4, 0, 0.1, 0.1, 0.1, 0, 0.1, 0.2, 0, 0.2, 0.1, 0, 0, 0.2, 0.4, 0.2, 0, 0.8, 0.6, 0.8, 1.0],
                                           'g': [0.8, 0.1, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                           'p': [0, 0.1, 0.2, 0.1, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0, 0, 0.3, 0.5, 0.5, 0.3, 0.7, 0.5, 0.5, 0.7],
                                           'z': [0.3, 0.2, 0.1, 0, 0, 0, 0.4, 0, 0, 0.4, 0, 0, 0, 0.4, 0.4, 0, 0, 0.6, 0.6, 1, 1]},
                                          index=index)

feature_probabilities_two = pd.DataFrame({'b': [0, 0.1, 0.2, 0.1, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0, 0, 0.3, 0.5, 0.5, 0.3, 0.7, 0.5, 0.5, 0.7],
                                           'g': [0.4, 0, 0.1, 0.1, 0.1, 0, 0.1, 0.2, 0, 0.2, 0.1, 0, 0, 0.2, 0.4, 0.2, 0, 0.8, 0.6, 0.8, 1.0],
                                           'p': [0.8, 0.1, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                           'z': [0.5, 0.3, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]},
                                          index=index)

def flesh_out(possibility_matrix, cards_per_player, known_cards):
    rows_good = True
    columns_good = True
    cards = []
    if (possibility_matrix.sum(axis=1) == 0).any():
        raise ValueError('Impossible matrix')
    for card, row in possibility_matrix.loc[possibility_matrix.sum(axis=1)==1].iterrows():
        rows_good = False
        pl = row[row==1].index[0]
        known_cards[pl].append(card)
        cards_per_player[pl] -= 1
        if cards_per_player[pl] < 0:
            raise ValueError('Impossible matrix')
        cards.append(card)
    possibility_matrix.drop(cards, inplace=True)
    players = []
    for pl in possibility_matrix.columns:
        if possibility_matrix[pl].sum() < cards_per_player[pl]:
            raise ValueError('impossible matrix')
        elif possibility_matrix[pl].sum() == cards_per_player[pl]:
            columns_good = False
            for card in possibility_matrix.loc[possibility_matrix[pl] == 1]:
                known_cards[pl].append(card)
                cards_per_player[pl] -= 1
            if cards_per_player[pl] != 0:
                raise ValueError('impossible matrix')
            players.append(pl)
    possibility_matrix.drop(players, axis=1, inplace=True)
    if (not rows_good) or (not columns_good):
        flesh_out(possibility_matrix, cards_per_player, known_cards)



def sample(possibility_matrix, cards_per_player, feature_possibilities):
    feature_possibilities = deepcopy(feature_possibilities)
    feature_possibilities = [(fp + 0.01)/1.01 for fp in feature_possibilities]# make everything possible
    players = list(range(len(cards_per_player)))
    known_cards = [[] for _ in range(len(cards_per_player))]
    suits = list(feature_possibilities[0].columns)
    suit_players = list(itertools.product(suits, players))
    while len(suit_players) > 0:
        rand_idx = random.randint(0, len(suit_players)-1)
        suit, pl = suit_players.pop(rand_idx)
        if pl not in possibility_matrix.columns:
            continue
        max_count = min(possibility_matrix.loc[possibility_matrix.index.str.contains(suit), pl].sum(), cards_per_player[pl])
        if max_count <= 0:
            continue
        min_count = cards_per_player[pl] - possibility_matrix[pl].count() + max_count
        short_suit_p = 0
        if min_count <= 0:
            short_suit_p = feature_possibilities[pl].loc['short_suited', suit]
        only_one_p = 0
        if min_count <= 1:
            only_one_p = feature_possibilities[pl].loc[feature_possibilities[pl].index.str.contains('only'), suit].sum()
        multiple_p = 0
        if max_count >= 2:
            multiple_p = feature_possibilities[pl].loc[feature_possibilities[pl].index.str.contains('highest'), suit].sum()
        if (short_suit_p + only_one_p + multiple_p) == 0:
            raise ValueError('shouldnt be possible')
        selection = random.choices(range(3), weights=[short_suit_p, only_one_p, multiple_p])[0]
        # possible = pd.Series(data=0, index=['{}{}'.format(suit, num) for num in range(1, 10)])
        possible = possibility_matrix.loc[possibility_matrix.index.str.contains(suit), pl]
        # possible[values.index] = values
        # possible = np.array(possibility_matrix.loc[possibility_matrix.index.str.contains(suit), pl])
        if selection in [0, 1]:
            possibility_matrix.loc[possibility_matrix.index.str.contains(suit), pl] = 0
        if selection == 1:
            probable = feature_possibilities[pl].loc[feature_possibilities[pl].index.str.contains('only'), suit]
            probable.index = ['{}{}'.format(suit, num) for num in range(1, 5)]
            weights = (probable*possible).dropna()
            card = random.choices(weights.index, weights=weights)
            possibility_matrix.loc[card, :] = 0
            possibility_matrix.loc[card, pl] = 1
        if selection == 2:
            probable = feature_possibilities[pl].loc[feature_possibilities[pl].index.str.contains('has'), suit]
            probable.index = ['{}{}'.format(suit, num) for num in range(1, 5)]
            weights = (probable*possible).dropna()
            cards = random.choices(weights.index, weights=weights, k=2)
            possibility_matrix.loc[cards, :] = 0
            possibility_matrix.loc[cards, pl] = 1
        flesh_out(possibility_matrix, cards_per_player, known_cards)
    while not possibility_matrix.empty:
        card = random.choices(possibility_matrix.index)[0]
        possible = np.array(possibility_matrix.loc[card])
        probs = np.array([feature_possibilities[pl].loc['has_{}'.format(card[1]), card[0]] for pl in possibility_matrix.columns])
        player = random.choices(possibility_matrix.columns, weights=possible*probs)
        possibility_matrix.loc[card, :] = 0
        possibility_matrix.loc[card, player] = 1
        flesh_out(possibility_matrix, cards_per_player, known_cards)
    return known_cards

s = sample(possibility_matrix, cards_per_player, [feature_probabilities_zero, feature_probabilities_one, feature_probabilities_two])

# t = copy(possibility_matrix)
# cpp = copy(cards_per_player)
# kc = [[], [], []]
# flesh_out(t, cards_per_player=cards_per_player, known_cards=kc)



