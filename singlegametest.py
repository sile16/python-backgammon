import time
import numpy as np
import unittest
from state import State, set_debug, set_random_seed, MoveSequence
        
start_time = time.time()
n_games = 10

for _ in range(n_games):
    state = State()
    state.pick_first_player()
    set_debug(False)
    
    max_moves = 0
    while not state.isTerminal():
        state.roll_dice()
        moves = state.get_legal_moves()
        mlen = len(moves)
        
        if mlen > max_moves:
            max_moves = mlen

        if mlen > 0:
            state.do_moves(moves[np.random.randint(mlen)])

        if state.isTerminal():
            print(f"Winner: {state.winner} Points: {state.points} NumMoves: {state.get_move_count()} max_moves: {max_moves}")
            # print the numpy array state.board
            print("Board state:")
            print(np.array_str(state.board, precision=2, suppress_small=True))
            break

duration = time.time() - start_time
games_per_second = n_games / duration
print(f"\nPerformance: {games_per_second:.2f} games/second")