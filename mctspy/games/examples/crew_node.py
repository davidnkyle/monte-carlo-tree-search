from collections import defaultdict

import numpy as np
import pandas as pd
from copy import copy, deepcopy






from mctspy.games.examples.crew_game_state import SUITS, DECK, CrewStatePublic
from mctspy.games.examples.integer_programming import initial_sample


def card_list_to_series(card_list):
    card_series = pd.Series(dtype=int)
    for pl in range(len(card_list)):
        for card in card_list[pl]:
            card_series[card] = pl
    return card_series


def card_series_to_list(card_series, num_players):
    card_list = [[] for _ in range(num_players)]
    for card, player in card_series.iteritems():
        card_list[player].append(card)
    return card_list

FEATURES = ['short_suited_{}'.format(suit) for suit in SUITS] + \
    ['only_{}'.format(c) for c in DECK] + \
    ['highest_{}'.format(c) for c in DECK] + \
    ['lowest_{}'.format(c) for c in DECK] + \
    ['has_{}'.format(c) for c in DECK] + \
    ['no_{}'.format(c) for c in DECK]
FEATURE_DICT_BASE = pd.Series(dtype=int)
for f in FEATURES:
    if ('has' not in f) and ('no' not in f):
        FEATURE_DICT_BASE[f] = 1
    else:
        FEATURE_DICT_BASE[f] = 0

class CooperativeGameNode():

    def __init__(self, state, unknown_hands, parent=None, parent_action=None, root=False, feature_weights=None):
        self.state = state
        self.parent = parent
        self.children = []
        # if unknown_hands is None:
        #     unknown_hands = [[] for _ in range(self.state.players)]
        self.unknown_hands = deepcopy(unknown_hands)
        self.parent_action = parent_action  # add a parameter for parent action
        self._n = pd.DataFrame(index=FEATURES, columns=range(self.state.players), data=0)
        # self._n = np.zeros((state.players, DECK_SIZE*2 + 5))
        # self._N = np.zeros((state.players, DECK_SIZE * 2 + 5))
        self._number_of_visits_total = 0
        self._number_of_wins = pd.DataFrame(index=FEATURES, columns=range(self.state.players), data=0)
        self._number_of_wins_total = 0
        self._results = defaultdict(int)
        self._all_untried_actions = None
        if feature_weights is None:
            feature_weights = FEATURE_DICT_BASE
        self.feature_weights = feature_weights
        self.root = root

    # @property
    def untried_actions(self):
        if self._all_untried_actions is None:
            self._all_untried_actions = self.state.get_all_actions()
        legal = self.state.get_legal_actions(self.unknown_hands[self.state.turn])
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

    def features_from_hand(self, unknown_hand, weights=False):
        """ makes pandas series with index of features
        if weights is True, only features with positive weight will be included
        and values will correspond to their weight
        if weights is False, all features of the unknown hand are in the index
        other features are NOT in the series.

        :param unknown_hand:
        :param weights:
        :return:
        """
        features = []
        hand = copy(unknown_hand)
        hand.sort()
        for suit in SUITS:
            subset = [c for c in hand if suit in c]
            if len(subset) == 0:
                features.append('short_suited_{}'.format(suit))
            elif len(subset) == 1:
                features.append('only_{}'.format(subset[0]))
            else:
                features.append('highest_{}'.format(subset[-1]))
                features.append('lowest_{}'.format(subset[0]))
                max_card = 4
                if suit == 'z':
                    max_card = 2
                for c in ['{}{}'.format(suit, n) for n in range(1, max_card + 1)]:
                    if c in subset:
                        features.append('has_{}'.format(c))
                    else:
                        features.append('no_{}'.format(c))
        if len(features) == 0:
            raise KeyError('every hand must have at least one feature')
        if weights:
            weighted = self.feature_weights[features]
            weighted = weighted[weighted > 0]
            if weighted.empty:
                raise KeyError('every hand must have at least one weighted feature')
            return weighted
        return pd.Series(data=1, index=features)

    def score(self, parent_hand, c_param):
        """ weighted score
        if hand has no features, then error
        if parent node has never been visited then error
        if some feature of the hand has ever been tried and c_param is nonzero, then inf is returned
        if no feature of the hand has ever been tried and c_param is zero, then 0 is returned

        :param unknown_hand:
        :param c_param:
        :return:
        """
        unknown_hand = self.state.unknown_hand(parent_hand)
        weights = self.features_from_hand(unknown_hand, weights=True)
        # weights = features.loc[self.n[self.state.turn] > 0] # only consider weights where n is positive
        n = self.n.loc[weights.index, self.parent.state.turn]
        score_per_feat = self.q.loc[weights.index, self.parent.state.turn] / n
        score_per_feat.loc[n == 0] = 0
        if c_param != 0:
            if self.parent._number_of_visits_total == 0:
                raise ZeroDivisionError('You cant score a node that whose parent has never been visited')
            exploration = np.log(self.parent._number_of_visits_total) / n
            score_per_feat = score_per_feat + c_param * exploration
        scores = score_per_feat * weights / weights.sum()
        return scores.sum()

    def best_child(self, hand, c_param=0.0, expand_if_necessary=False):
        """ determine the best move from this node given the hand provided
        if the node has no children with from the valid actions, an error is thrown
        otherwise the child with the max score (no exploration constant) is chosen

        :param hand:
        :return: child node
        """
        if self.state.is_game_over():
            raise ValueError('dont ask for the next best move if the game is already over')
        uhand = self.state.unknown_hand(hand)
        legal_actions = self.state.get_legal_actions(uhand)
        valid_children = [c for c in self.children if c.parent_action in legal_actions]
        if len(valid_children) == 0:
            if expand_if_necessary:
                action = legal_actions[0]
                next_state = self.state.move(action)
                sample = initial_sample(possible_matrix_df=next_state.possible_cards, player_totals=next_state.num_unknown)
                unknown_hands = card_series_to_list(sample, self.state.players)
                child_node = CooperativeGameNode(
                    next_state, unknown_hands=unknown_hands, parent=self, parent_action=action
                )
                self.children.append(child_node)
                return child_node
            raise ValueError('This node is a leaf!')
        best_child = max(valid_children, key=lambda x: x.score(c_param=c_param, parent_hand=uhand))
        return best_child

    def select(self):
        if not self.is_fully_expanded():
            raise ValueError('Only select a node when it is already fully expanded')
        node = self.best_child(hand=self.unknown_hands[self.state.turn], c_param=1.4)
        new_hands = [node.state.unknown_hand(h) for h in self.unknown_hands]
        node.unknown_hands = new_hands
        return node

    #
    # def best_child(self, unknown_hand, c_param=1.4):
    #
    #     best_score = -np.inf
    #     best_child = None
    #     for c in self.children:
    #         turn = self.state.turn
    #         c_unkown_hand = list(set(unknown_hand).difference(c.state.known_cards))
    #         full_hand = c.state.known_hands[turn] + c_unkown_hand
    #         if c.parent_action in self.state.get_legal_actions(full_hand):
    #             if best_child is None:
    #                 best_child = c
    #             features = features_from_hand(c_unkown_hand)
    #             if c_param != 0:
    #                 rel_features = [f for f in features if self.state.feature_weights[f] == 1]
    #                 if c.n[rel_features, turn] == 0:
    #                     return c
    #             exploration = np.sqrt((2 * np.log(self._number_of_visits_total) / c._number_of_visits_total))
    #             weights = c.state.feature_weights[features]
    #             weights = weights/weights.sum()
    #             score = (c.q[features, turn]/c.n[features, turn]*weights).sum() + c_param*exploration
    #             # vector = np.divide(c.q[rel_features, turn], b, out=np.zeros_like(b), where=b != 0) + c_param * exploration
    #             # score = np.average(vector, weights=[self.state.feature_weights[f] for f in rel_features])
    #             if score > best_score:
    #                 best_child = c
    #                 best_score = score
    #
    #     if best_child is None:
    #         return self.expand(self.state.known_hands[self.state.turn] + unknown_hand)
    #     return best_child

    def expand(self):
        action = self.untried_actions()[0]
        self._all_untried_actions.remove(action)
        next_state = self.state.move(action)
        valid_hands = [next_state.unknown_hand(h) for h in self.unknown_hands]
        child_node = CooperativeGameNode(
            next_state, unknown_hands=valid_hands, parent=self, parent_action=action
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        unknown_hands = deepcopy(self.unknown_hands)
        # game_states = []
        # unknown_hand_states = []
        while not current_rollout_state.is_game_over():
            # game_states.append(current_rollout_state)
            # unknown_hand_states.append(unknown_hands)
            turn = current_rollout_state.turn
            hand = unknown_hands[current_rollout_state.turn]
            possible_moves = current_rollout_state.get_legal_actions(hand)
            action = self.rollout_policy(possible_moves)
            # select_phase = current_rollout_state.select_goals_phase
            # com_phase = current_rollout_state.communication_phase
            for idx in range(current_rollout_state.players):
                if len(unknown_hands[idx]) != current_rollout_state.num_unknown[idx]:
                    raise ValueError('uh oh')
            current_rollout_state = current_rollout_state.move(action)
            unknown_hands = [current_rollout_state.unknown_hand(h) for h in unknown_hands]

        return current_rollout_state.game_result

    def feature_df(self):
        all_features = pd.DataFrame(index=FEATURES)
        for idx in range(self.state.players):
            all_features[idx] = self.features_from_hand(self.unknown_hands[idx])
        all_features = all_features.fillna(0)
        return all_features

    def backpropagate(self, result):
        all_features = self.feature_df()
        self._n += all_features
        self._number_of_wins += all_features * result
        # if self.parent and not self.parent.state.select_goals_phase and not self.parent.state.communication_phase:
        #     turn = self.parent.state.turn
        #     idx = DECK.index(self.parent_action)
        #     n_matrix[turn, idx] = 1
        #     n_matrix[turn, idx+DECK_SIZE] = 0
        #     # update the short suited attribute
        #     n_matrix[turn, 2*DECK_SIZE:] = [n_matrix[turn, DECK_SIZE+j*9:DECK_SIZE + min((j+1)*9, DECK_SIZE)].min() for j in range(len(SUITS))]
        # self._n += n_matrix
        self._number_of_visits_total += 1
        self._number_of_wins_total += result
        if self.parent and not self.root:
            self.parent.backpropagate(result)

    def rollout_policy(self, possible_moves):
        next_move = np.random.randint(len(possible_moves))
        # print(next_move)
        return possible_moves[next_move%len(possible_moves)]

    def relative_frequency(self):
        cards = self.state.possible_cards.index
        players = self.state.possible_cards.columns
        onlies = self.n.loc[['only_{}'.format(c) for c in cards], players]
        multis = self.n.loc[['has_{}'.format(c) for c in cards], players]
        rf = pd.DataFrame(index=cards, columns=players, data=onlies.values + multis.values)
        result = self.state.possible_cards
        return result

if __name__ == '__main__':

    game = CrewStatePublic(players=3, goal_cards=['g3', 'p1', 'b4'], captain=1)
    weights = FEATURE_DICT_BASE
    for goal in game.goal_cards:
        weights['has_{}'.format(goal)] = 1
        weights['no_{}'.format(goal)] = 1
    uhands = [['b1', 'g2', 'g3', 'p4'], ['g1', 'g4', 'p2', 'p3'], ['b2', 'b3', 'b4', 'p1', 'z1']]
    root = CooperativeGameNode(game, unknown_hands=uhands, root=True, feature_weights=weights)
    node = root.expand()
    result = node.rollout()
    node.backpropagate(result)
    bc = root.best_child(uhands[1])
    score = root.children[0].score(uhands[1])
    uhands1 = [['b1', 'g2', 'p1', 'z1'], ['g1', 'g4',  'g3', 'p4'], ['b2', 'b3', 'b4', 'p2', 'p3']]
    root.unknown_hands = uhands1
    node1 = root.expand()
    result1 = node1.rollout()
    node1.backpropagate(result1)
    bc1 = root.best_child(uhands1[1])
