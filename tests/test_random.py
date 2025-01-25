import pytest
import numpy as np
import os
import sys

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the build/lib directory to Python's path
build_lib_dir = os.path.join(project_root, 'build', 'lib')
if build_lib_dir not in sys.path:
    sys.path.insert(0, build_lib_dir)


from bg_game import set_debug, get_debug, BGGame
from bg_moves import MoveSequence

# Constants for player colors
WHITE = 0
BLACK = 1
NONE = 2

def test_state_initialization():
    """Test the initialization of the game state."""
    bg = BGGame()
    bg.set_player(WHITE)
    bg.randomize_seed()
    #bg.set_seed(42)
    # by deafult we should seed random and have different resutls


    unseeded_rolls = [(2, 2), (6, 3), (5, 3), (1, 3), (6, 2), (1, 6), (1, 3), (4, 6), (2, 2), (5, 5)]
    rolls = []

    found_diff = False
    found_diff_from_d1 = False

    for i in range(10):
        bg.roll_dice()
        assert bg.dice[0] > 0 and bg.dice[0] < 7, "Dice roll should be between 1 and 6"
        assert bg.dice[1] > 0 and bg.dice[1] < 7, "Dice roll should be between 1 and 6"
        rolls.append(bg.dice)
        if bg.dice != unseeded_rolls[i]:
            found_diff = True
        if bg.dice != rolls[0]:
            found_diff_from_d1 = True
    
    assert found_diff, "Dice rolls should be different with different seeds"
    assert found_diff_from_d1, "Dice rolls should be different from the first roll"
    
    

    
    