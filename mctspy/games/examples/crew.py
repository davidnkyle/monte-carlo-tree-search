import time
from collections import defaultdict
from copy import copy, deepcopy

import numpy as np
import random

import pandas as pd

from mctspy.games.examples.crew_game_state import SUITS, DECK, DECK_SIZE, CrewStatePublic



random.seed(336)

# implement communication signal

from mctspy.tree.nodes import MonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch

EPSILON = 0.000000000001




class MCTSCrew():
    def __init__(self, node, card_matrix):
        self.root = node
        self.card_matrix = card_matrix
        self.hands = None


    def create_tree(self, simulations_number=None, total_simulation_seconds=None):
        if simulations_number is None:
            assert (total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while True:
                attempts = 20
                while attempts > 0:
                    try:
                        sample = sample_from_matrix(self.card_matrix, self.root.state.num_cards_per_player)
                        break
                    except ValueError:
                        attempts -= 1
                self.hands = [DECK_ARRAY[np.where(sample[pl, :])] for pl in range(self.card_matrix.shape[0])]
                v = self._tree_policy()
                reward = v.rollout(hands=self.hands)
                n_matrix = np.array([v.state.get_feature_vector(hand) for hand in self.hands])
                # n_matrix = np.zeros((v.state.players, DECK_SIZE))
                # for pl in range(v.state.players):
                #     n_matrix[pl, [DECK.index(c) for c in self.hands[pl]]] = 1
                v.backpropagate(reward, n_matrix=n_matrix)
                if time.time() > end_time:
                    break
        else:
            for _ in range(0, simulations_number):
                attempts = 20
                while attempts > 0:
                    try:
                        sample = sample_from_matrix(self.card_matrix, self.root.state.num_cards_per_player)
                        break
                    except ValueError:
                        attempts -= 1
                if abs(sample.sum() - self.card_matrix.sum()) > EPSILON:
                    raise ValueError('Hey now')
                self.hands = [list(DECK_ARRAY[np.where(sample[pl, :])]) for pl in range(self.card_matrix.shape[0])]
                v = self._tree_policy()
                reward = v.rollout(hands=self.hands)
                n_matrix = np.array([v.state.get_feature_vector(hand) for hand in self.hands])
                # n_matrix = np.zeros((v.state.players, DECK_SIZE))
                # for pl in range(v.state.players):
                #     n_matrix[pl, [DECK.index(c) for c in self.hands[pl]]] = 1
                v.backpropagate(reward, n_matrix=n_matrix)
        # to select best child go for exploitation only
        # return self.root.best_child(c_param=0.)

    def best_action(self, hand):
        return self.root.best_child(hand=hand, c_param=0.0)


    def _tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            hand = self.hands[current_node.state.turn]
            if not current_node.is_fully_expanded(hand):
                new_node = current_node.expand(hand)
                if not current_node.state.select_goals_phase and not current_node.state.communication_phase:
                    hand.remove(new_node.parent_action)
                return new_node
            else:
                select_phase = current_node.state.select_goals_phase
                com_phase = current_node.state.communication_phase
                current_node = current_node.best_child(hand)
                if not select_phase and not com_phase:
                    hand.remove(current_node.parent_action)
        return current_node


if __name__ == '__main__':
    results = []
    for _ in range(2):
        players = 3
        deck = copy(DECK)
        random.shuffle(deck)
        initial_hands = [deck[(i * DECK_SIZE) // players:((i + 1) * DECK_SIZE) // players] for i in range(players)]
        true_hands = deepcopy(initial_hands)
        captain = ['z4' in hand for hand in initial_hands].index(True)
        num_goals = 3
        # goals = [[] for _ in range(players)]
        goal_deck = copy(DECK[0:-4])
        random.shuffle(goal_deck)
        initial_board_state = CrewStatePublic(players=players, num_goals=num_goals, goal_cards=goal_deck[0:num_goals], captain=captain)
        hand_size = np.array([len(h) for h in initial_hands])
        hand_size[captain] -= 1
        cm = np.ones((players, DECK_SIZE))*hand_size[:,None]/(DECK_SIZE-1)
        cm[:, DECK_SIZE-1] = 0
        cm[captain, DECK_SIZE-1] = 1
        board_state = initial_board_state
        rounds_left = 13
        print('goal cards: {}'.format(board_state.goal_cards))
        print(true_hands)
        print()
        print('leading: {}'.format(board_state.leading))
        root = CooperativeGameNode(state=board_state, root=True)
        while board_state.game_result is None:
            mcts = MCTSCrew(root, card_matrix=cm)
            simulations = 5000
            mcts.create_tree(simulations)
            best_node = mcts.best_action(true_hands[board_state.turn])
            den = best_node._number_of_visits_total
            if den == 0:
                den = 1
            nm = best_node.n[:, :DECK_SIZE] / den
            if not root.state.select_goals_phase and not root.state.communication_phase:
                true_hands[board_state.turn].remove(best_node.parent_action)
                act_idx = DECK.index(best_node.parent_action)
                nm[:, act_idx] = 0
            if abs(nm.sum()-round(nm.sum())) > EPSILON:
                raise ValueError('Hey now')
            possible = best_node.state.possible_cards
            row_total = possible.sum(axis=0)
            possible = np.divide(possible, row_total, out=np.zeros_like(possible), where=row_total != 0)
            cm = 0.99*nm + 0.01*possible
            if best_node.state.rounds_left < rounds_left:
                print(board_state.coms)
                print(board_state.trick + [best_node.parent_action])
                print(best_node.state.goals)
                print(true_hands)
                print()
                print('leading: {}'.format(best_node.state.leading))
                rounds_left = best_node.state.rounds_left
            board_state = best_node.state
            best_node.root = True
            root = best_node
        print('result {}'.format(board_state.game_result))
        print('rounds left {}'.format(board_state.rounds_left))
        results.append(board_state.game_result)
    print(sum(results))



