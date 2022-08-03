import time
from collections import defaultdict
from copy import copy, deepcopy

import numpy as np
import random

from mctspy.tree.nodes import MonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch

EPSILON = 0.000000000001
SUITS = 'bgpyz'
DECK = ['{}{}'.format(color, number) for color in 'bgpy' for number in range(1, 10)] + \
    ['z{}'.format(number) for number in range(1, 5)]
DECK_ARRAY = np.array(DECK)
DECK_SIZE = len(DECK)

# see https://arxiv.org/pdf/1012.0256.pdf
def select_with_probs(v, w, m):
    if m == 0:
        return np.array([])
    if len(v) != len(w):
        raise ValueError('weights and values are unequal')
    if m > len(v):
        raise ValueError('m must be smaller than length of v')
    # scale weights if not scaled already
    w = w/w.sum()*m
    # update overweight items
    w[w>1] = 1
    selection = v[0:m]
    # total_weight = sum(w[0:m])
    for k in range(m, len(v)):
        # total_weight += w[k]
        p = w[k]
        if random.random() < p:
            selection[random.randint(0, m-1)] = v[k]
    return selection

def sample1_from_matrix(matrix):
    matrix = copy(matrix)
    new = np.zeros(matrix.shape)
    while matrix.max() > 0:
        idx = matrix.max(axis=0).argmax()
        player = random.choices(range(matrix.shape[0]), weights=matrix[:, idx])[0]
        new[player, idx] = 1
        matrix[:, idx] = 0
        a = np.ceil(matrix.sum(axis=1))
        b = matrix.sum(axis=1)
        matrix = matrix*np.divide(a, b, out=np.zeros_like(a), where=b != 0)[:, None]
        matrix[matrix > 1] = 1
        leftover = 1 - matrix.sum(axis=0) + matrix[player, :]
        leftover[matrix[player, :] < EPSILON] = 0
        matrix[player, :] = leftover
    return matrix


def sample_from_matrix(matrix):
    matrix = copy(matrix)
    new = np.zeros(matrix.shape)
    goals = matrix.sum(axis=1)
    while matrix.max() > 0:
        idx = matrix.max(axis=0).argmax()
        player = random.choices(range(matrix.shape[0]), weights=matrix[:, idx])[0]
        new[player, idx] = 1
        matrix[:, idx] = 0
        if new[player, :].sum() > goals[player] - EPSILON:
            matrix[player, :] = 0
            b = matrix.sum(axis=0)
            matrix = np.divide(matrix, b, out=np.zeros_like(matrix), where=b != 0)
    return new



def evaluate_trick(trick):
    if len(trick) == 0:
        raise ValueError('No trick to evaluate')
    suit = trick[0][0]
    cards = [c for c in trick if c[0] in (suit, 'z')]
    cards.sort(reverse=True)
    return cards[0]


class CrewStatePublic():
    def __init__(self, players=3, num_goals=3, captain=None):
        if (players < 3) or (players > 5):
            raise ValueError('Only allow between 3 and 5 players')
        self.players = players
        self.discard = []
        # self.possible_hands = np.ones((DECK_SIZE, self.players))
        self.captain = captain
        # if self.captain is not None:
        #     self.possible_hands[-1,:] = 0
        #     self.possible_hands[-1,self.captain] = 1
        self.leading = 0
        self.turn = 0
        self.num_goals = num_goals
        self.goals = [[] for _ in range(self.players)]
        self.rounds_left = DECK_SIZE//self.players
        self.trick = []

    def _deal_goals(self):
        cards = random.sample(DECK, self.num_goals)
        self.goals = [[] for _ in range(self.players)]
        i = self.captain
        for c in cards:
            self.goals[i].append(c)
            i = (i+1)%self.players

    @property
    def game_result(self):
        for pl in self.goals:
            for c in pl:
                if c in self.discard:
                    return 0 # if the goal is still active and in the discard pile, there is no way to win
        if all([len(g) == 0 for g in self.goals]):
            return 1
        if self.rounds_left == 0:
            return 0
        return None

    def is_game_over(self):
        return self.game_result is not None

    def move(self, move):
        new = deepcopy(self)
        # new.hands[new.turn].remove(move)
        # new.possible_hands[DECK.index(move), :] = 0 # nobody has the card that was just played
        # if len(self.trick) > 0 and self.trick[0][0] != move[0]: # short suited
        #     suit_id = SUITS.index(self.trick[0][0])
        #     new.possible_hands[suit_id*9:(suit_id+1)*9, self.turn] = 0
        new.trick.append(move)
        if len(new.trick) < new.players:
            new.turn = (new.turn + 1) % self.players
            return new
        winner = (evaluate_trick(new.trick) + new.leading) % new.players
        new.goals[winner] = list(set(new.goals[winner]).difference(new.trick)) # remove any goals in the trick
        new.discard += new.trick # add trick to discard
        new.trick = []
        new.leading = winner
        new.turn = winner
        new.rounds_left -= 1
        return new

    def is_move_legal(self, move, hand):
        if not move in hand: # you dont have this card in your hand
            return False
        if len(self.trick) > 0:
            leading_suit = self.trick[0][0] # you must follow suit if you can
            if leading_suit in [c[0] for c in hand]:
                return move[0] == leading_suit
        return True

    def get_legal_actions(self, hand):
        return [c for c in hand if self.is_move_legal(c, hand)]



class CooperativeGameNode():

    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.parent_action = parent_action # add a parameter for parent action
        self._number_of_visits = np.zeros((state.players, DECK_SIZE))
        self._number_of_wins = np.zeros((state.players, DECK_SIZE))
        self._results = defaultdict(int)
        self._untried_actions = None

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self):
        return self._number_of_wins

    @property
    def n(self):
        return self._number_of_visits

    def is_fully_expanded(self, hand):
        # changed to incorperate the current hand
        return len(set(hand).intersection(self.untried_actions)) == 0

    def best_child(self,  hand, c_param=1.4):
        hand_vector = np.zeros(DECK_SIZE)
        for c in hand:
            hand_vector[DECK.index(c)] = 1/len(hand)
        choices_weights = [
            np.dot(hand_vector, (c.q[self.state.turn, :] / c.n[self.state.turn, :]) + c_param * np.sqrt((2 * np.log(self.n[self.state.turn]) / c.n[c.state.turn])))
            for c in self.children if c.parent_action in hand
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, hand):
        # edited to only consider the current hand
        current_untried = [x for x in self.untried_actions if x in hand]
        if len(current_untried) == 0:
            raise ValueError('You should only call this function if there is something to expand')
        action = current_untried[0]
        self.untried_actions.remove(action)
        # action = self.untried_actions.pop() # this removes it from _untried_actions attribute
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
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions(hand=hands[current_rollout_state.turn])
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result

    def backpropagate(self, result, n_matrix):
        self._number_of_visits += n_matrix
        self._number_of_wins += n_matrix*result
        if self.parent:
            n_matrix[self.parent.turn, DECK.index(self.parent_action)] = 1
            self.parent.backpropagate(result, n_matrix)

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]



class MCTSCrew(MonteCarloTreeSearch):
    def __init__(self, node, card_matrix):
        super().__init__(node)
        self.card_matrix = card_matrix


    def create_tree(self, simulations_number=None, total_simulation_seconds=None):
        if simulations_number is None:
            assert (total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while True:
                sample = sample_from_matrix(self.card_matrix)
                hands = [DECK_ARRAY[np.where(sample[pl, :])] for pl in range(self.card_matrix.shape[0])]
                v = self._tree_policy(hands=hands)
                reward = v.rollout(hands=hands)
                v.backpropagate(reward, n_matrix=np.zeros((v.state.players, DECK_SIZE)))
                if time.time() > end_time:
                    break
        else:
            for _ in range(0, simulations_number):
                sample = sample_from_matrix(self.card_matrix)
                hands = [DECK_ARRAY[np.where(sample[pl, :])] for pl in range(self.card_matrix.shape[0])]
                v = self._tree_policy(hands=hands)
                reward = v.rollout(hands=hands)
                v.backpropagate(reward, n_matrix=np.zeros((v.state.players, DECK_SIZE)))
        # to select best child go for exploitation only
        # return self.root.best_child(c_param=0.)

    def best_action(self, hand):
        return self.root.best_child(hand=hand, c_param=0.0)

    # def _generate_player_hands(self):
    #     # todo allow for different hand sizes
    #     new_card_matrix = copy(self.card_matrix)
    #     sample = sample_from_matrix(new_card_matrix)
    #     hands = []
    #     for pl in range(sample.shape[1]):
    #         hands.append(sample)



    def _tree_policy(self, hands):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded(hand=hands[current_node.turn]):
                return current_node.expand(hand=hands[current_node.turn])
            else:
                current_node = current_node.best_child(hand=hands[current_node.turn])
        return current_node

#
# class TrueCrewState(CrewState):
#     def __init__(self, players=3, num_goals=3):
#         super().__init__(players=players, num_goals=num_goals)
#
#         # deal cards and determine captain
#         self._deal_cards()
#         self.discard = []
#         self.captain = 0
#         self.leading = 0
#         self.turn = 0
#         for pl in range(self.players):
#             if 'b4' in self.hands[pl]:
#                 self.captain = pl
#                 break
#         self.leading = self.captain
#         self.turn = self.captain
#
#         # deal goals
#         self.num_goals = num_goals
#         self._deal_goals()
#
#         # other attributes
#         self.rounds_left = DECK_SIZE//self.players
#         self.trick = []
#
#     def _deal_cards(self):
#         deck = copy(DECK)
#         random.shuffle(deck)
#         self.hands = [deck[(i * DECK_SIZE) // self.players:((i + 1) * DECK_SIZE) // self.players] for i in range(self.players)]
#
#     def is_move_legal(self, move):
#         hand = self.hands[self.turn]
#         if not move in hand: # you dont have this card in your hand
#             return False
#         if len(self.trick) > 0:
#             leading_suit = self.trick[0][0] # you must follow suit if you can
#             if leading_suit in [c[0] for c in hand]:
#                 return move[0] == leading_suit
#         return True
#
#     def move(self, move):
#         if not self.is_move_legal(move):
#             raise ValueError('Illegal move')
#         new = self.move(move)
#         new.hands[self.turn].remove(move)
#         return new
#
#     def get_legal_actions(self):
#         return self.hands[self.turn]
#
#
#
# class CrewStatePublic(CrewState):
#     def __init__(self, players=3, num_goals=3):
#         super().__init__(players=players, num_goals=num_goals)
#         self.possible_hands = np.ones((DECK_SIZE, self.players))
#

if __name__ == '__main__':
    players = 3
    initial_board_state = CrewStatePublic()
    initial_hands = [[], [], []] # todo update this
    true_hands = deepcopy(initial_hands)
    # todo change this to be an actual starting card matrix
    cm = np.ones((players, DECK_SIZE)) * 13 / DECK_SIZE
    board_state = initial_board_state
    while board_state.game_result is None:
        root = CooperativeGameNode(state=board_state)
        mcts = MCTSCrew(root, card_matrix=cm)
        simulations = 10000
        mcts.create_tree(simulations)
        best_node = mcts.best_action(true_hands[board_state.turn])
        true_hands[board_state.turn].remove(best_node.parent_action)
        cm = best_node.n/simulations
        board_state = best_node.state
        print(board_state.board)


