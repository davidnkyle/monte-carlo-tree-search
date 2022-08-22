from copy import deepcopy, copy
import numpy as np
from crew_game_state import CrewStatePublic, DECK, DECK_SIZE
import pandas as pd


def rollout(game_state, iterations):
    current_rollout_state = game_state
    game_states = []
    for _ in range(iterations):
        while not current_rollout_state.is_game_over():
            game_states.append(current_rollout_state)
            possible_moves = current_rollout_state.get_legal_actions([])
            action = possible_moves.pop(np.random.randint(len(possible_moves)))
            current_rollout_state = current_rollout_state.move(action)
            game_states.append(game_state)
        goal_completion = 1- (current_rollout_state.goals_remaining / game_state.num_goals)
        if goal_completion == 1:
            print(_)
            return True
        index = int((len(game_states)*goal_completion)//1)
        current_rollout_state = game_states[index]
        game_states = game_states[0:index]
    return False

if __name__ == '__main__':
    for seed in range(100, 110):
        np.random.seed(seed)
        players = 5
        deck = copy(DECK)
        np.random.shuffle(deck)
        initial_hands = [deck[(i * DECK_SIZE) // players:((i + 1) * DECK_SIZE) // players] for i in range(players)]
        true_hands = deepcopy(initial_hands)
        captain = [DECK[-1] in hand for hand in initial_hands].index(True)
        num_goals = 3
        goal_deck = copy(DECK[0:-4])
        np.random.shuffle(goal_deck)
        initial_board_state = CrewStatePublic(players=players, num_goals=num_goals, goal_cards=goal_deck[0:num_goals],
                                              captain=captain)
        initial_board_state.known_hands = true_hands
        initial_board_state.possible_cards = pd.DataFrame()
        result = rollout(initial_board_state, 1000)
        print(result)


