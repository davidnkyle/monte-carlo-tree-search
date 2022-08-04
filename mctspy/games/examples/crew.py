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
    # todo try random selection of values instead of starting with max
    # but still take care of the ones first
    # how does the distribution compare?
    matrix = copy(matrix)
    new = np.zeros(matrix.shape)
    goals = matrix.sum(axis=1)
    while matrix.max() > 0:
        if matrix.max() > 1 - EPSILON:
            idx = matrix.max(axis=0).argmax()
        else:
            idx = random.choices(range(matrix.shape[1]), weights=matrix.sum(axis=0))
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

    def get_all_actions(self):
        return copy(DECK)



class CooperativeGameNode():

    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.parent_action = parent_action # add a parameter for parent action
        self._number_of_visits = np.zeros((state.players, DECK_SIZE))
        self._number_of_wins = np.zeros((state.players, DECK_SIZE))
        self._results = defaultdict(int)
        self._all_untried_actions = None

    # @property
    def untried_actions(self, hand):
        if self._all_untried_actions is None:
            self._all_untried_actions = self.state.get_all_actions()
        legal = self.state.get_legal_actions(hand)
        return [a for a in self._all_untried_actions if a in legal]

    @property
    def q(self):
        return self._number_of_wins

    @property
    def n(self):
        return self._number_of_visits

    def is_fully_expanded(self, hand):
        return len(self.untried_actions(hand)) == 0

    def best_child(self,  hand, c_param=1.4):
        hand_vector = np.zeros(DECK_SIZE)
        hand_indices = []
        for c in hand:
            hand_indices.append(DECK.index(c))
            hand_vector[DECK.index(c)] = 1/len(hand)
        best_score = 0
        best_child = self.children[0]
        for c in self.children:
            hnd = copy(hand_indices)
            hnd.remove(c.parent_action)
            turn = self.state.turn
            if c.parent_action in self.state.get_legal_actions():
                if c.n[turn, hnd].min() == 0 or self.n[turn, hnd].min() == 0:
                    return c
                score = (c.q[turn, hnd] / c.n[turn, hnd]) + c_param * \
                        np.sqrt((2 * np.log(self.n[turn, hnd]) / c.n[turn, hnd])).mean()
                if score > best_score:
                    best_child = c
                    best_score = score
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



class MCTSCrew():
    def __init__(self, node, card_matrix):
        self.root = node
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


    def _tree_policy(self, hands):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded(hands[current_node.turn]):
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node


if __name__ == '__main__':
    random.seed(329)
    players = 3
    num_goals = 3
    deck = copy(DECK)
    random.shuffle(deck)
    initial_hands = [deck[(i * DECK_SIZE) // players:((i + 1) * DECK_SIZE) // players] for i in range(players)]
    true_hands = deepcopy(initial_hands)
    captain = ['z4' in hand for hand in initial_hands].index(True)
    initial_board_state = CrewStatePublic(players=players, num_goals=num_goals, captain=captain)
    hand_size = np.array([len(h) for h in initial_hands])
    hand_size[captain] -= 1
    cm = np.ones((players, DECK_SIZE))*hand_size/(DECK_SIZE-1)
    cm[:, DECK_SIZE-1] = 0
    cm[captain, DECK_SIZE-1] = 1
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
    # matrix = np.array([[0.4, 0.6, 0.4, 0.0, 0.3, 0.3, 0],
    #                    [0.6, 0.4, 0.0, 0.0, 0.7, 0.3, 0],
    #                    [0.0, 0.0, 0.6, 1.0, 0.0, 0.4, 0]])
    # totals = np.zeros((3, 7))
    # for _ in range(10000):
    #     new = sample_from_matrix(matrix)
    #     totals += new


