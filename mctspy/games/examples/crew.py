import time
from collections import defaultdict
from copy import copy, deepcopy

import numpy as np
import random

import pandas as pd

from mctspy.games.examples.crew_game_state import SUITS, DECK, DECK_SIZE, CrewStatePublic, FEATURES
from mctspy.games.examples.permutation3 import features_from_hand



random.seed(336)

# implement communication signal

from mctspy.tree.nodes import MonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch

EPSILON = 0.000000000001


class CooperativeGameNode():

    def __init__(self, state, unknown_hands, valid_sample, parent=None, parent_action=None, root=False):
        self.state = state
        self.parent = parent
        self.children = []
        self.unknown_hands = deepcopy(unknown_hands)
        self.valid_sample = valid_sample
        self.parent_action = parent_action # add a parameter for parent action
        self._n = pd.DataFrame(index=FEATURES, columns=range(self.state.players), data=0)
        # self._n = np.zeros((state.players, DECK_SIZE*2 + 5))
        # self._N = np.zeros((state.players, DECK_SIZE * 2 + 5))
        self._number_of_visits_total = 0
        self._number_of_wins = pd.DataFrame(index=FEATURES, columns=range(self.state.players), data=0)
        self._number_of_wins_total = 0
        self._results = defaultdict(int)
        self._all_untried_actions = None
        self.root = root

    # @property
    def untried_actions(self):
        if self._all_untried_actions is None:
            self._all_untried_actions = self.state.get_all_actions()
        legal = self.state.get_legal_actions(self.hand)
        return [a for a in self._all_untried_actions if a in legal]

    @property
    def hand(self):
        turn = self.state.turn
        return self.state.known_hands[turn] + self.unknown_hands[turn]

    @property
    def q(self):
        return self._number_of_wins

    @property
    def n(self):
        return self._n

    def is_fully_expanded(self):
        return len(self.untried_actions()) == 0


    def best_child(self, unknown_hand, c_param=1.4):

        best_score = -np.inf
        best_child = None
        for c in self.children:
            turn = self.state.turn
            c_unkown_hand = list(set(unknown_hand).difference(c.state.known_hands[turn]))
            full_hand = c.state.known_hands[turn] + c_unkown_hand
            if c.parent_action in self.state.get_legal_actions(full_hand):
                if best_child is None:
                    best_child = c
                features = features_from_hand(c_unkown_hand)
                if c_param != 0:
                    rel_features = [f for f in features if self.state.feature_weights[f] == 1]
                    if c.n[rel_features, turn] == 0:
                        return c
                exploration = np.sqrt((2 * np.log(self._number_of_visits_total) / c._number_of_visits_total))
                weights = c.state.feature_weights[features]
                weights = weights/weights.sum()
                score = (c.q[features, turn]/c.n[features, turn]*weights).sum() + c_param*exploration
                # vector = np.divide(c.q[rel_features, turn], b, out=np.zeros_like(b), where=b != 0) + c_param * exploration
                # score = np.average(vector, weights=[self.state.feature_weights[f] for f in rel_features])
                if score > best_score:
                    best_child = c
                    best_score = score

        if best_child is None:
            return self.expand(self.state.known_hands[self.state.turn] + unknown_hand)
        return best_child

    def expand(self, hand):
        action = self.untried_actions(hand)[0]
        self._all_untried_actions.remove(action)
        next_state = self.state.move(action)
        child_node = CooperativeGameNode(
            next_state, parent=self, parent_action=action
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self, hands):
        current_rollout_state = self.state
        hands = deepcopy(hands)
        while not current_rollout_state.is_game_over():
            hand = hands[current_rollout_state.turn]
            possible_moves = current_rollout_state.get_legal_actions(hand=hand)
            action = self.rollout_policy(possible_moves)
            select_phase = current_rollout_state.select_goals_phase
            com_phase = current_rollout_state.communication_phase
            current_rollout_state = current_rollout_state.move(action)
            if not select_phase and not com_phase:
                hand.remove(action)
        return current_rollout_state.game_result

    def backpropagate(self, result, n_matrix):
        self._N += n_matrix
        if self.parent and not self.parent.state.select_goals_phase and not self.parent.state.communication_phase:
            turn = self.parent.state.turn
            idx = DECK.index(self.parent_action)
            n_matrix[turn, idx] = 1
            n_matrix[turn, idx+DECK_SIZE] = 0
            # update the short suited attribute
            n_matrix[turn, 2*DECK_SIZE:] = [n_matrix[turn, DECK_SIZE+j*9:DECK_SIZE + min((j+1)*9, DECK_SIZE)].min() for j in range(len(SUITS))]
        self._n += n_matrix
        self._number_of_visits_total += 1
        self._number_of_wins += n_matrix*result
        self._number_of_wins_total += result
        if self.parent and not self.root:
            self.parent.backpropagate(result, n_matrix)

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]



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



