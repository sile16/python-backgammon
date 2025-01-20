
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


from bg_game import set_debug, get_debug, set_random_seed, BGGame
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
    count = 5
    for x in range(count):
        default_state.reset()
        default_state.set_player(WHITE)
    
        while not default_state.isTerminal():
            default_state.roll_dice()
            moves = default_state.get_legal_moves2()
            moves2 = default_state.get_legal_moves()
            assert(len(moves) == len(moves2))
            
            move = moves[np.random.randint(len(moves))]
            
            #print(f"applying dice roll {move.dice}")
            default_state.do_moves(move)
         
        
            
            
    
        
    
    

if __name__ == "__main__":
    pytest.main()