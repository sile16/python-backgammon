# verify_installation.py
# run 

import sys
import os

# Optionally, ensure the virtual environment's site-packages are in sys.path
# This is usually handled by pip during installation

from bg_game import set_debug, get_debug, BGGame
from bg_moves import MoveSequence

def main():
    set_debug(False)
    game = BGGame()
    game.set_player(0)
    game.randomize_seed()
    game.roll_dice()
    print("BGGame initialized successfully.")

if __name__ == "__main__":
    main()
