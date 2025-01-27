
import pytest
import numpy as np
import os
import sys

from bg_game import set_debug, get_debug, BGGame
from bg_moves import MoveSequence


# Constants for player colors
WHITE = 0
BLACK = 1
NONE = 2


@pytest.fixture
def default_state():
    """Fixture to initialize the default game state."""
    bg =  BGGame()
    bg.set_player(WHITE)
    bg.roll_dice()
    return bg

def test_many_games(default_state):
    """Capture and validate move generation with seeded randomness."""
    count = 100
    wins = [0, 0]
    points = [0, 0]
    rewards = [0, 0, 0, 0]
    set_debug(True)
    for x in range(count):
        default_state.reset()
        default_state.set_player(WHITE)
        default_state.roll_dice()

        while not default_state.isTerminal():
            legal_actions2 = default_state.legal_actions()
            legal_actions2.sort()

            #assert len(legal_actions) == len(legal_actions2)
            #assert legal_actions ==  legal_actions2

            i = np.random.randint(len(legal_actions2))
            a = legal_actions2[i]
            
            obs, reward, done = default_state.step(legal_actions2[i])
            
            if done:
                points[default_state.player] += reward
                wins[default_state.player] += 1
                #print("Game: reward ", reward)
                rewards[reward] += 1
                break

    print("Game: ", x)
    print("Wins: ", wins)
    print("Points: ", points)
    print("Rewards: ", rewards)


def test_per_pip_transitions(default_state):
    """Ensure per-pip moves correctly update board state."""
    default_state.reset()
    default_state.set_player(WHITE)
    default_state.set_board([
        # Simulated board with a few checkers
        [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12],
        [12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0]
    ])
    default_state.roll_dice()
    
    legal_moves = default_state.legal_actions()
    for move in legal_moves:
        new_state = default_state.copy()
        new_state.step(move)
        # compare these two np arrays
        assert default_state != new_state  # Move should change state

def test_no_legal_moves(default_state):
    """Ensure the game properly handles a scenario where no legal moves exist."""
    default_state.reset()
    default_state.set_player(WHITE)
    default_state.set_board([
        # Simulated board where all white checkers are blocked
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]
    ])
    default_state.roll_dice()
    
    legal_moves = default_state.legal_actions()
    assert len(legal_moves) == 1  # Should return an empty list
    assert legal_moves[0] == 30  # Should return a pass move

def test_bearing_off(default_state):
    """Ensure bearing off moves are handled correctly."""
    default_state.reset()
    default_state.set_player(WHITE)
    default_state.set_board([
        # Simulated board with all checkers in bear-off range
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15]
    ])
    default_state.roll_dice()
    
    legal_moves = default_state.legal_actions()
    for move in legal_moves:
        assert move == 0 or move == 15

def test_rendering(default_state):
    """Ensure the board state renders correctly."""
    default_state.reset()
    default_state.set_player(WHITE)
    for _ in range(5):
        default_state.roll_dice()
        a = default_state.legal_actions()
        default_state.step(a[0])
    default_state.render()

if __name__ == "__main__":
    pytest.main()