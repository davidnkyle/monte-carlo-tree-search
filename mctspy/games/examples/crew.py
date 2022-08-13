import time
from collections import defaultdict
from copy import copy, deepcopy

import numpy as np
import random

import pandas as pd

from mctspy.games.examples.crew_game_state import SUITS, DECK, DECK_SIZE, CrewStatePublic
from mctspy.games.examples.crew_node import FEATURE_DICT_BASE, CooperativeGameNode
from mctspy.games.examples.permutation3 import swap

random.seed(336)

# implement communication signal

from mctspy.tree.nodes import MonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch

EPSILON = 0.000000000001




class MCTSCrew():
    def __init__(self, node):
        self.root = node


    def card_list_to_series(self, card_list):
        card_series = pd.Series(dtype=int)
        for pl in range(len(card_list)):
            for card in card_list[pl]:
                card_series[card] = pl
        return card_series

    def card_series_to_list(self, card_series, num_players):
        card_list = [[] for _ in range(num_players)]
        for card, player in card_series.iteritems():
            card_list[player].append(card)
        return card_list


    def create_tree(self, card_matrix, simulations_number=None, total_simulation_seconds=None, sample_seed=None):
        if simulations_number is None:
            assert (total_simulation_seconds is not None)
            # end_time = time.time() + total_simulation_seconds
            # while True:
            #     attempts = 20
            #     while attempts > 0:
            #         try:
            #             sample = sample_from_matrix(self.card_matrix, self.root.state.num_cards_per_player)
            #             break
            #         except ValueError:
            #             attempts -= 1
            #     self.hands = [DECK_ARRAY[np.where(sample[pl, :])] for pl in range(self.card_matrix.shape[0])]
            #     v = self._tree_policy()
            #     reward = v.rollout(hands=self.hands)
            #     n_matrix = np.array([v.state.get_feature_vector(hand) for hand in self.hands])
            #     # n_matrix = np.zeros((v.state.players, DECK_SIZE))
            #     # for pl in range(v.state.players):
            #     #     n_matrix[pl, [DECK.index(c) for c in self.hands[pl]]] = 1
            #     v.backpropagate(reward, n_matrix=n_matrix)
            #     if time.time() > end_time:
            #         break
        else:
            # create the starting deck as a series with card indices and player index as values
            if sample_seed is None:
                sample_seed = self.root.unknown_cards
            starting_deck = self.card_list_to_series(sample_seed)
            sample = deepcopy(starting_deck)
            for _ in range(0, simulations_number):
                sample = swap(starting_deck=sample, rf=card_matrix, N=20)
                unknown_hands = self.card_series_to_list(sample, self.root.state.players)
                root.unknown_hands = unknown_hands
                # self.hands = [list(DECK_ARRAY[np.where(sample[pl, :])]) for pl in range(self.card_matrix.shape[0])]
                v = self._tree_policy()
                reward = v.rollout()
                # n_matrix = np.array([v.state.get_feature_vector(hand) for hand in self.hands])
                # n_matrix = np.zeros((v.state.players, DECK_SIZE))
                # for pl in range(v.state.players):
                #     n_matrix[pl, [DECK.index(c) for c in self.hands[pl]]] = 1
                v.backpropagate(reward)
        # to select best child go for exploitation only
        # return self.root.best_child(c_param=0.)

    def best_action(self, hand):
        return self.root.best_child(hand=hand, c_param=0.0)


    def _tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                new_node = current_node.expand()
                return new_node
            else:
                current_node = current_node.select()
        return current_node


if __name__ == '__main__':
    players = 3
    deck = copy(DECK)
    np.random.shuffle(deck)
    initial_hands = [deck[(i * DECK_SIZE) // players:((i + 1) * DECK_SIZE) // players] for i in range(players)]
    true_hands = deepcopy(initial_hands)
    captain = [DECK[-1] in hand for hand in initial_hands].index(True)
    unknown_hands = deepcopy(true_hands)
    unknown_hands[captain].remove(DECK[-1])
    num_goals = 3
    goal_deck = copy(DECK[0:-2])
    np.random.shuffle(goal_deck)
    initial_board_state = CrewStatePublic(players=players, num_goals=num_goals, goal_cards=goal_deck[0:num_goals], captain=captain)
    weights = FEATURE_DICT_BASE
    for goal in initial_board_state.goal_cards:
        weights['has_{}'.format(goal)] = 1
        weights['no_{}'.format(goal)] = 1
    cm = initial_board_state.possible_cards.copy()
    board_state = initial_board_state
    rounds_left = initial_board_state.rounds_left
    print('goal cards: {}'.format(board_state.goal_cards))
    print(unknown_hands)
    print()
    print('leading: {}'.format(board_state.leading))
    first_seed = []
    deck_idx = 0
    for pl in range(initial_board_state.players):
        first_seed.append(DECK[deck_idx:deck_idx+initial_board_state.num_unknown[pl]])
        deck_idx += initial_board_state.num_unknown[pl]
    root = CooperativeGameNode(initial_board_state, unknown_hands=first_seed, root=True, feature_weights=weights)
    while board_state.game_result is None:
        mcts = MCTSCrew(root)
        simulations = 100
        mcts.create_tree(card_matrix=cm, simulations_number=simulations)
        best_node = mcts.best_action(unknown_hands[board_state.turn])
        unknown_hands = [board_state.unknown_hand(h) for h in unknown_hands]
        cm = best_node.n + best_node.state.possible_cards
        # den = best_node._number_of_visits_total
        # if den == 0:
        #     den = 1
        # nm = best_node.n[:, :DECK_SIZE] / den
        # if not root.state.select_goals_phase and not root.state.communication_phase:
        #     true_hands[board_state.turn].remove(best_node.parent_action)
        #     act_idx = DECK.index(best_node.parent_action)
        #     nm[:, act_idx] = 0
        # if abs(nm.sum()-round(nm.sum())) > EPSILON:
        #     raise ValueError('Hey now')
        # possible = best_node.state.possible_cards
        # row_total = possible.sum(axis=0)
        # possible = np.divide(possible, row_total, out=np.zeros_like(possible), where=row_total != 0)
        # cm = 0.99*nm + 0.01*possible
        if best_node.state.rounds_left < rounds_left:
            print(board_state.coms)
            print(board_state.trick + [best_node.parent_action])
            print(best_node.state.goals)
            print(best_node.state.known_hands)
            print(unknown_hands)
            print()
            print('leading: {}'.format(best_node.state.leading))
            rounds_left = best_node.state.rounds_left
        board_state = best_node.state
        best_node.root = True
        root = best_node
    print('result {}'.format(board_state.game_result))
    print('rounds left {}'.format(board_state.rounds_left))



