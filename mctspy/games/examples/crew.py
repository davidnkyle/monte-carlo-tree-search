from copy import copy, deepcopy

import numpy as np
import random

DECK = ['{}{}'.format(color, number) for color in 'pbyg' for number in range(1, 10)] + \
    ['z{}'.format(number) for number in range(1, 5)]
DECK_SIZE = len(DECK)

def evaluate_trick(trick):
    if len(trick) == 0:
        raise ValueError('No trick to evaluate')
    suit = trick[0][0]
    cards = [c for c in trick if c[0] in (suit, 'z')]
    cards.sort(reverse=True)
    return cards[0]


class TrueCrewState():

    def __init__(self, players=3, num_goals=3):
        if (players < 3) or (players > 5):
            raise ValueError('Only allow between 3 and 5 players')
        self.players=players

        # deal cards and determine captain
        self._deal_cards()
        self.discard = []
        self.captain = 0
        self.leading = 0
        self.turn = 0
        for pl in range(self.players):
            if 'b4' in self.hands[pl]:
                self.captain = pl
                break
        self.leading = self.captain
        self.turn = self.captain

        # deal goals
        self.num_goals = num_goals
        self._deal_goals()

        # other attributes
        self.rounds_left = DECK_SIZE//self.players
        self.trick = []

    def _deal_cards(self):
        deck = copy(DECK)
        random.shuffle(deck)
        self.hands = [deck[(i * DECK_SIZE) // self.players:((i + 1) * DECK_SIZE) // self.players] for i in range(self.players)]

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

    def is_move_legal(self, move):
        hand = self.hands[self.turn]
        if not move in hand: # you dont have this card in your hand
            return False
        if len(self.trick) > 0:
            leading_suit = self.trick[0][0] # you must follow suit if you can
            if leading_suit in [c[0] for c in hand]:
                return move[0] == leading_suit
        return True


    def move(self, move):
        if not self.is_move_legal(move):
            raise ValueError('Illegal move')
        new = deepcopy(self)
        new.hands[new.turn].remove(move)
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

    def get_legal_actions(self):
        return self.hands[self.turn]


