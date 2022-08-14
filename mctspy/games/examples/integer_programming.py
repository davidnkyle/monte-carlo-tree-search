
import cvxpy as cp
import numpy as np
import pandas as pd




def initial_sample(possible_matrix_df, player_totals):

    if possible_matrix_df.empty:
        return pd.Series(dtype=int)

    possible_matrix = possible_matrix_df.values
    numpy_totals = [player_totals[idx] for idx in possible_matrix_df.columns]
    players = possible_matrix.shape[1]
    cards = possible_matrix.shape[0]
    hand_constraints = [[] for _ in range(players)]
    card_constraints = [[] for _ in range(cards)]

    var_size = 0
    for j in range(players):
        for i in range(cards):
            if possible_matrix[i, j] == 1:
                var_size += 1
                for pl in range(players):
                    if pl == j:
                        hand_constraints[pl].append(1)
                    else:
                        hand_constraints[pl].append(0)
                for card in range(cards):
                    if card == i:
                        card_constraints[card].append(1)
                    else:
                        card_constraints[card].append(0)

    A = np.array(hand_constraints + card_constraints)
    b = np.array(numpy_totals + [1 for _ in range(cards)])


    x = cp.Variable(shape=var_size, boolean=True)
    constraints = [cp.matmul(A, x) == b]
    objective = cp.Minimize(cp.sum(x))
    prob = cp.Problem(objective, constraints=constraints)
    _ = prob.solve()

    # print(solution)
    # print(x.value)

    sample = [-1 for _ in range(cards)]
    idx = 0
    for j in range(players):
        for i in range(cards):
            if possible_matrix[i, j] == 1:
                if x.value[idx] == 1:
                    sample[i] = possible_matrix_df.columns[j]
                idx += 1

    result = pd.Series(index=possible_matrix_df.index, data=sample)
    return result


if __name__ == '__main__':
    player_totals = [3, 3, 3]

    possible_matrix_df = pd.DataFrame(data=[[0, 1, 1],
                                            [0, 1, 1],
                                            [1, 0, 1],
                                            [0, 1, 1],
                                            [1, 1, 1],
                                            [1, 1, 1],
                                            [1, 1, 0],
                                            [1, 1, 0],
                                            [1, 1, 0]], index=['b1', 'b2', 'b3', 'g1', 'g3', 'g4', 'p1', 'p2', 'z1'],
                                      columns=[0, 1, 2])
    result = initial_sample(possible_matrix_df=possible_matrix_df, player_totals=player_totals)
