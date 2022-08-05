import time
from collections import defaultdict
from copy import copy, deepcopy

import numpy as np
import random

random.seed(334)

from mctspy.tree.nodes import MonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch

EPSILON = 0.000000000001
SUITS = 'bgpyz'
DECK = ['{}{}'.format(color, number) for color in 'bgpy' for number in range(1, 10)] + \
    ['z{}'.format(number) for number in range(1, 5)]
DECK_ARRAY = np.array(DECK)
DECK_SIZE = len(DECK)

def sample_from_matrix(matrix, cards_per_player):
    total = sum(cards_per_player)
    matrix = copy(matrix)
    new = np.zeros(matrix.shape)
    # cards_per_player = matrix.sum(axis=1)
    while matrix.max() > 0:
        non_zeros = np.count_nonzero(matrix, axis=0)
        num = np.min(non_zeros[np.nonzero(non_zeros)])
        idx = random.choices(np.where(non_zeros==num)[0])[0]
        player = random.choices(range(matrix.shape[0]), weights=matrix[:, idx])[0]
        new[player, idx] = 1
        matrix[:, idx] = 0
        if new[player, :].sum() > cards_per_player[player] - EPSILON:
            if matrix[player, :].max() > 1 - EPSILON:
                raise ValueError('impossible sample')
            matrix[player, :] = 0
            b = matrix.sum(axis=0)
            matrix = np.divide(matrix, b, out=np.zeros_like(matrix), where=b != 0)
    if abs(new.sum() - total) > EPSILON:
        raise ValueError('Uh oh')
    return new



def evaluate_trick(trick):
    if len(trick) == 0:
        raise ValueError('No trick to evaluate')
    suit = trick[0][0]
    cards = [c for c in trick if c[0] in (suit, 'z')]
    cards.sort(reverse=True)
    return trick.index(cards[0])


class CrewStatePublic():
    def __init__(self, players=3, num_goals=3, goals=None, captain=None):
        if (players < 3) or (players > 5):
            raise ValueError('Only allow between 3 and 5 players')
        self.players = players
        self.discard = []
        # self.possible_hands = np.ones((DECK_SIZE, self.players))
        self.captain = captain
        # if self.captain is not None:
        #     self.possible_hands[-1,:] = 0
        #     self.possible_hands[-1,self.captain] = 1
        self.leading = captain
        self.turn = captain
        self.num_goals = num_goals
        if goals is None:
            goals = [[] for _ in range(self.players)]
        self.goals = goals
        self.rounds_left = DECK_SIZE//self.players
        self.trick = []
        self.num_cards_per_player = [self.rounds_left for _ in range(self.players)]
        if self.players == 3:
            self.num_cards_per_player[2] += 1
        self.possible_cards = np.ones((self.players, DECK_SIZE))
        self.possible_cards[:, DECK.index('z4')] = 0
        self.possible_cards[self.captain, DECK.index('z4')] = 1


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
        new.trick.append(move)
        new.num_cards_per_player[self.turn] -= 1
        new.possible_cards[:, DECK.index(move)] = 0  # nobody has the card that was just played
        if len(new.trick) > 1:
            # if player did not follow suit they dont have any of that suit
            leading_suit = new.trick[0][0]
            if move[0] != leading_suit:
                start = SUITS.index(leading_suit)
                new.possible_cards[self.turn, start*9:(start+1)*9] = 0
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

    def __init__(self, state, parent=None, parent_action=None, root=False):
        self.state = state
        self.parent = parent
        self.children = []
        self.parent_action = parent_action # add a parameter for parent action
        self._number_of_visits = np.zeros((state.players, DECK_SIZE))
        self._number_of_wins = np.zeros((state.players, DECK_SIZE))
        self._results = defaultdict(int)
        self._all_untried_actions = None
        self.root = root

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
        hand_indices = [DECK.index(c) for c in hand]
        best_score = -np.inf
        best_child = None
        for c in self.children:
            hnd = copy(hand_indices)
            turn = self.state.turn
            if c.parent_action in self.state.get_legal_actions(hand):
                if c_param != 0:
                    if c.n[turn, hnd].min() == 0 or self.n[turn, hnd].min() == 0:
                        return c
                    exploration = np.sqrt((2 * np.log(self.n[turn, hnd]) / c.n[turn, hnd]))
                else:
                    exploration = 0
                b = c.n[turn, hnd]
                score = (np.divide(c.q[turn, hnd], b, out=np.zeros_like(b), where=b != 0) + c_param * exploration).mean()
                if score > best_score:
                    best_child = c
                    best_score = score
        if best_child is None:
            return self.expand(hand)
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
            current_rollout_state = current_rollout_state.move(action)
            hand.remove(action)
        return current_rollout_state.game_result

    def backpropagate(self, result, n_matrix):
        if self.parent and not self.root:
            n_matrix[self.parent.state.turn, DECK.index(self.parent_action)] = 1
            self._number_of_visits += n_matrix
            self._number_of_wins += n_matrix*result
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
                n_matrix = np.zeros((v.state.players, DECK_SIZE))
                for pl in range(v.state.players):
                    n_matrix[pl, [DECK.index(c) for c in self.hands[pl]]] = 1
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
                n_matrix = np.zeros((v.state.players, DECK_SIZE))
                for pl in range(v.state.players):
                    n_matrix[pl, [DECK.index(c) for c in self.hands[pl]]] = 1
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
                hand.remove(new_node.parent_action)
                return new_node
            else:
                current_node = current_node.best_child(hand)
                hand.remove(current_node.parent_action)
        return current_node


if __name__ == '__main__':

    players = 3
    deck = copy(DECK)
    random.shuffle(deck)
    initial_hands = [deck[(i * DECK_SIZE) // players:((i + 1) * DECK_SIZE) // players] for i in range(players)]
    true_hands = deepcopy(initial_hands)
    captain = ['z4' in hand for hand in initial_hands].index(True)
    num_goals = 3
    goals = [[] for _ in range(players)]
    goal_deck = copy(DECK[0:-4])
    random.shuffle(goal_deck)
    for idx in range(captain, captain + num_goals):
        goals[idx % players].append(goal_deck.pop(0))
    initial_board_state = CrewStatePublic(players=players, num_goals=num_goals, goals=goals, captain=captain)
    hand_size = np.array([len(h) for h in initial_hands])
    hand_size[captain] -= 1
    cm = np.ones((players, DECK_SIZE))*hand_size[:,None]/(DECK_SIZE-1)
    cm[:, DECK_SIZE-1] = 0
    cm[captain, DECK_SIZE-1] = 1
    board_state = initial_board_state
    rounds_left = 13
    print('leading: {}'.format(board_state.leading))
    root = CooperativeGameNode(state=board_state, root=True)
    while board_state.game_result is None:
        mcts = MCTSCrew(root, card_matrix=cm)
        simulations = 5000
        mcts.create_tree(simulations)
        best_node = mcts.best_action(true_hands[board_state.turn])
        true_hands[board_state.turn].remove(best_node.parent_action)
        act_idx = DECK.index(best_node.parent_action)
        # print(best_node.n)
        nm = best_node.n/best_node.n[board_state.turn, act_idx]
        nm[:, act_idx] = 0
        if abs(nm.sum()-round(nm.sum())) > EPSILON:
            raise ValueError('Hey now')
        possible = best_node.state.possible_cards
        row_total = possible.sum(axis=0)
        possible = np.divide(possible, row_total, out=np.zeros_like(possible), where=row_total != 0)
        cm = 0.99*nm + 0.01*possible
        if best_node.state.rounds_left < rounds_left:
            print(board_state.trick + [best_node.parent_action])
            print(best_node.state.goals)
            print()
            print('leading: {}'.format(best_node.state.leading))
            rounds_left = best_node.state.rounds_left
        board_state = best_node.state
        best_node.root = True
        root = best_node
    print('result {}'.format(board_state.game_result))
    print('rounds left {}'.format(board_state.rounds_left))



