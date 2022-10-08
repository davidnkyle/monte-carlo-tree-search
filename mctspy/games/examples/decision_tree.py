import time
from copy import copy, deepcopy

import pandas as pd

from mctspy.games.examples.crew_game_state import DECK, DECK_SIZE, CrewStatePublic, SUITS, map_known_cards
import numpy as np
import pickle

def hand_to_vector(hand):
    values = []
    for card in DECK:
        if card in hand:
            values.append(1)
        else:
            values.append(0)
    return values



def features_from_game_state(game_state):
    features = []
    suit_sums = [0, 0, 0, 0, 0]
    for pl in range(3):
        vec = hand_to_vector(game_state.known_hands[pl])
        features += vec
        for i in range(5):
            total = sum(vec[9 * i: 9 * (i + 1)])
            suit_sums[i] += total
            features.append(total)
        features.append(sum(vec))
        goal_vec = hand_to_vector(game_state.goals[pl])
        features += goal_vec
        features.append(sum(goal_vec))
    features += suit_sums
    features.append(num_goals)
    return features


def check_for_viability(turns, game_state, model):
    if turns == 1:
        for action in game_state.get_legal_actions([]):
            state = game_state.move(action)
            gr = state.game_result
            if gr is None:
                new = state.to_feature_form()
                if model.predict(np.array([features_from_game_state(new)]))[0] == 1:
                    return 1
            elif gr == 1:
                return 1
        return 0
    for action in game_state.get_legal_actions([]):
        state = game_state.move(action)
        if check_for_viability(turns-1, state, model) == 1:
            return 1
    return 0

if __name__ == '__main__':
    startTime = time.time()

    feature_cols = ['leading_{}'.format(c) for c in DECK] + ['leading_{}_total'.format(s) for s in SUITS] + ['leading_total_cards'] + \
                   ['leading_goal_{}'.format(c) for c in DECK] + ['leading_total_goals'] + \
                   ['pl1_{}'.format(c) for c in DECK] + ['leading_{}_total'.format(s) for s in SUITS] + ['pl1_total_cards'] + \
                   ['pl1_goal_{}'.format(c) for c in DECK] + ['pl1_total_goals'] + \
                   ['pl2_{}'.format(c) for c in DECK] + ['leading_{}_total'.format(s) for s in SUITS] + ['pl2_total_cards'] + \
                   ['pl2_goal_{}'.format(c) for c in DECK] + ['pl2_total_goals'] + \
                   ['{}_total'.format(s) for s in SUITS] + ['total_goals']

    with open(r'model_3pl_round13.pkl', 'rb') as f:
        pl3_round13_model = pickle.load(f)

    deck = copy(DECK)
    goal_deck = copy(DECK[0:-4])
    inputs = []
    results = []
    for seed in range(1000000, 3000000):
        np.random.seed(seed)
        players = 3
        np.random.shuffle(deck)

        # captain = [DECK[-1] in hand for hand in initial_hands].index(True)
        num_goals = 1
        initial_board_state = CrewStatePublic(players=players, num_goals=num_goals, goal_cards=[],
                                              captain=0)

        initial_board_state.leading = 0
        initial_board_state.num_unknown = [0, 0, 0]
        initial_board_state.rounds_left = 2
        initial_board_state.select_goals_phase = False
        all_cards_raw = deck[0:7]
        map = map_known_cards(all_cards_raw)
        all_cards = [map[c] for c in all_cards_raw]
        non_trump_cards = [c for c in all_cards if 'z' not in c]
        if len(non_trump_cards) == 0:
            continue
        num_goals = np.random.randint(len(non_trump_cards))+1
        goals = non_trump_cards[:num_goals]
        players_with_goals = np.random.choice(range(3), 2)
        for goal in goals:
            coin_flip = np.random.randint(2)
            initial_board_state.goals[players_with_goals[coin_flip]].append(goal)
        np.random.shuffle(all_cards)
        initial_hands = [all_cards[0:2], all_cards[2:4], all_cards[4:6]]
        player_with_2_cards = np.random.randint(3)
        initial_hands[player_with_2_cards].append(all_cards[-1])
        initial_board_state.known_hands = initial_hands
        initial_board_state.possible_cards = pd.DataFrame()
        features = features_from_game_state(initial_board_state)
        inputs.append(features)
        result = check_for_viability(3, initial_board_state, pl3_round13_model)
        # result = 0
        # for idx in range(2):
        #     if player_with_2_cards == 0:
        #         board = initial_board_state.move(initial_hands[0][idx])
        #     else:
        #         board = initial_board_state.move(initial_hands[0][0])
        #     if player_with_2_cards == 1:
        #         board = board.move(initial_hands[1][idx])
        #     else:
        #         board = board.move(initial_hands[1][0])
        #     if player_with_2_cards == 2:
        #         board = board.move(initial_hands[2][idx])
        #     else:
        #         board = board.move(initial_hands[2][0])
        #     result = max(result, board.game_result)
        results.append(result)

    df = pd.DataFrame(data=inputs, columns=feature_cols)
    df['result'] = results
    df.to_csv('pl3_round12_2000000_20220822.csv')
    executionTime = (time.time() - startTime) / 60 / 60
    print('Execution time in hours: ' + str(executionTime))
