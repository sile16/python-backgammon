import pytest
import numpy as np
import os
import sys
import time

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


@pytest.fixture
def default_state():
    """Fixture to initialize the default game state."""
    return BGGame()



# Debug Mode Tests
def test_set_debug():
    set_debug(True)
    assert get_debug() is True

    set_debug(False)
    assert get_debug() is False

# Random Seed Tests
def test_set_random_seed():

    state = BGGame()
    
    state.randomize_seed(42)
    state.roll_dice()
    d1 = state.dice
    
    state.randomize_seed(42)
    state.roll_dice()
    d2 = state.dice

    assert d1 == d2

    state.randomize_seed(43)
    state.roll_dice()
    d2 = state.dice

    assert d1 != d2

def test_move_sequence_encode_decode():

    ms = MoveSequence()

    i1 = ms.toIndex()
    assert i1 == 0

    ms2 = MoveSequence.toSequenceFromIndex(0)
    assert ms == ms2


    for i in range(100):
        ms.n_moves = 0

        for j in range(np.random.randint(1, 3)):
            ms.add_move(  np.random.randint(0, 25), np.random.randint(1, 7) )
        
        i1 = ms.toIndex()
        ms2 = MoveSequence.toSequenceFromIndex(i1)
        i2 = ms2.toIndex()
        assert(i1 == i2)
        assert(ms == ms2)

        
        for i in range(1, 7):
            ms.n_moves = 0
            for j in range(np.random.randint(1, 5)):  #number of rolls
                src = np.random.randint(0, 25)
                ms.add_move(src, i)
        
            i1 = ms.toIndex()
            ms2 = MoveSequence.toSequenceFromIndex(i1)
            i2 = ms2.toIndex()
            assert(i1 == i2)
            assert(ms == ms2)
        

def test_randomness():
    # run a bunch of dice rolls and make sure they are not all the same
    state = BGGame()
    state.set_player(WHITE)

    #create a dice histogram
    dice_hist = {}
    for i in range(1, 7):
        for j in range(1, 7):
            dice_hist[(i, j)] = 0
    
    for i in range(10000):
        state.roll_dice()
        dice_hist[(state.dice[0], state.dice[1])] += 1
    
    #check that the % distribution is within 1% of 1/36
    for k, v in dice_hist.items():
        assert abs(v/10000 - 1/36) < 0.01


# State Initialization Tests
def test_state_initialization(default_state):
    """Verify that the State object initializes correctly."""
    assert default_state is not None
    assert isinstance(default_state, BGGame)

# Dice Roll Operations
def test_dice_roll():
    """Verify dice roll logic."""
    state = BGGame()
    state.roll_dice()
    dice = (2, 5)
    state.set_dice(dice)
    assert state.dice == (2, 5)

    dice_rolls = []
    for _ in range(10):
        state.roll_dice()
        assert 1 <= state.dice[0] <= 6
        assert 1 <= state.dice[1] <= 6
        dice_rolls.append(tuple(state.dice))

    dice_set = set(dice_rolls)
    assert len(dice_set) > 2  # Ensure randomness

# Move Generation Tests
def test_generate_valid_moves(default_state):
    """Verify move generation logic."""
    
    default_state.set_player(WHITE)
    default_state.roll_dice()

    valid_moves = default_state.get_legal_moves()
    valid_moves2 = default_state.get_legal_moves2()
    assert isinstance(valid_moves, list)
    assert isinstance(valid_moves2, list)
    for moveset in [valid_moves, valid_moves2]:
        for move in moveset:
            for i in range(move.n_moves):
                assert move.moves[i]['src'] >= 0
                assert move.moves[i]['src'] < 25
                assert move.moves[i]['n'] > 0
                assert move.moves[i]['n'] < 7
            
def test_largest_die(default_state):
    

    orig = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14],
        [9, 0, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int8)
    default_state.set_board(orig.copy())

    default_state.set_player(WHITE)
    default_state.set_dice((3, 4))

    valid_moves = default_state.get_legal_moves()
    valid_moves2 = default_state.get_legal_moves2()
    assert isinstance(valid_moves, list)
    assert isinstance(valid_moves2, list)

    assert len(valid_moves) == 1
    assert len(valid_moves2) == 1
    for moveSeq in [valid_moves[0], valid_moves2[0]]:
        assert moveSeq.n_moves == 1
        assert moveSeq.moves[0]['n'] == 4


# Edge Case Tests
def test_no_valid_moves(default_state):
    """Test a board state where no moves are possible."""
    #set_debug(True)
    default_state.set_player(WHITE)
    default_state.board[0, :] = 0  # Empty white's board
    default_state.board[0, 0] = 15  # Place all checkers on the bar

    default_state.set_dice((6,6))
    valid_moves = default_state.get_legal_moves()
    valid_moves2 = default_state.get_legal_moves2()

    assert len(valid_moves) == 1
    assert len(valid_moves) == 1

    assert valid_moves[0].n_moves == 0
    assert valid_moves2[0].n_moves == 0
    assert valid_moves[0].moves[0]['n'] == 0
    assert valid_moves2[0].moves[0]['n'] == 0

# Boundary Conditions
def test_bar_and_bear_off_logic(default_state):
    """Test moves from bar and bear-off positions."""

    default_state.board[0, 0] = 2  # Place checkers on bar
    default_state.board[0, 1] = 0  # Place checkers on bear-off
    default_state.set_player(WHITE)
    default_state.set_dice((3, 4))

    valid_moves = default_state.get_legal_moves()
    valid_moves2 = default_state.get_legal_moves2()
    assert len(valid_moves) == 1
    assert len(valid_moves2) == 1
    for moveSeq in [valid_moves[0], valid_moves2[0]]:
        assert moveSeq.n_moves == 2
        assert moveSeq.moves[0]['src'] == 0
        assert moveSeq.moves[1]['src'] == 0
        assert moveSeq.moves[0]['n'] + moveSeq.moves[1]['n'] == 7

    default_state.board[0, :] = 0  # Remove checkers from bar
    default_state.board[0, 24] = 15  # All checkers ready to bear off
    assert default_state.can_bear_off()

# Regression Tests
def test_seeded_move_generation(default_state):
    """Capture and validate move generation with seeded randomness."""
    default_state.randomize_seed(42)
    default_state.set_player(WHITE)
    default_state.set_dice((6, 1))

    moves = default_state.get_legal_moves()
    default_state.randomize_seed(42)
    repeated_moves = default_state.get_legal_moves()


    assert moves == repeated_moves

def test_seeded_move_generation(default_state):
    """Capture and validate move generation with seeded randomness."""
    default_state.randomize_seed(42)
    default_state.set_player(WHITE)
    default_state.set_dice((6, 1))

    moves = default_state.get_legal_moves()
    default_state.randomize_seed(42)
    repeated_moves = default_state.get_legal_moves()

    assert moves == repeated_moves

def test_many_moves(default_state):
    """Capture and validate move generation with seeded randomness."""

    #set_debug(True)
    default_state.set_player(WHITE)
    default_state.set_dice((3, 3))
    board = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 3, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 5]
    ], dtype=np.int8)

    default_state.set_board(board)
    moves = default_state.get_legal_moves()
    moves2 = default_state.get_legal_moves2()
    #assert moves == moves2
    assert len(moves) == 368
    assert len(moves2) == 368

    default_state.reset()
    default_state.set_player(WHITE)
    default_state.set_dice((1, 1))
    board = np.array([
        [0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 5]
    ], dtype=np.int8)

    default_state.set_board(board)
    moves = default_state.get_legal_moves()
    moves2 = default_state.get_legal_moves2()
    #assert moves == moves2
    assert len(moves) == 1259
    assert len(moves2) == 1259

    default_state.reset()
    default_state.set_player(WHITE)
    default_state.set_dice((2, 1))
    board = np.array([
        [0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 5]
    ], dtype=np.int8)

    default_state.set_board(board)
    moves = default_state.get_legal_moves()
    moves2 = default_state.get_legal_moves2()
    #assert moves == moves2
    assert len(moves) == 161
    assert len(moves2) == 161 # todo why?

    s1 = set()
    s2 = set()
    for m in moves:
        s1.add(m.final_board.tobytes())

    for m in moves2:
        s2.add(m.final_board.tobytes())
        if m.final_board.tobytes() not in s1:
            print(m.final_board)
            print(m.moves)
    
    for m in moves:
        if m.final_board.tobytes() not in s2:
            print(m.final_board)
            print(m.moves)
    
def test_roll_dice_unique(default_state):
    default_state.set_player(WHITE)
    default_state.roll_dice()
    dice1 = default_state.dice
    default_state.roll_dice()
    dice2 = default_state.dice
    assert dice1 != dice2

def test_both_players_on_bar(default_state):

    board = np.array(
        [[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 2, 2, 5, 0, 0],
         [0, 7, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.int8)
    
    default_state.set_board(board)
    default_state.set_player(WHITE)
    default_state.set_dice((4, 2))
    moves = default_state.get_legal_moves()
    moves2 = default_state.get_legal_moves2()
    assert len(moves) == len(moves2)

def test_dice_index(default_state):
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            i = default_state.diceToIndex(d1, d2)
            new_dice = default_state.indexToDice(i)
            assert d1 in new_dice
            assert d2 in new_dice

def test_get_all_moves_all_dice(default_state):
    default_state.set_player(WHITE)
    default_state.set_dice((3, 4))
    moves = default_state.get_all_moves_all_dice()
    assert len(moves) == 21
    

def test_many_games(default_state):
    """Capture and validate move generation with seeded randomness."""
    count = 3000
    white_win = 0
    black_win = 0
    normal = 0
    gammon = 0
    backgammon = 0
    games = 0
    steps = 0
    delta = 0


    start_time = time.time()
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
            steps += 1
        games += 1
        if default_state.winner == WHITE:
            white_win += 1
        elif default_state.winner == BLACK:
            black_win += 1

        if default_state.points == 1:
            normal += 1
        elif default_state.points == 2:
            gammon += 1
        elif default_state.points == 3:
            backgammon += 1
        
    end_time = time.time()
    delta = end_time - start_time
    
    print(f"White Wins: {white_win} Black Wins: {black_win} Normal: {normal} Gammon: {gammon} Backgammon: {backgammon}")
    print(f"Total Games: {games} Total Steps: {steps} Total Time: {end_time - start_time}")
    print(f"Total Games/s: {games/delta} Total Steps/s: {steps/delta} Total Time: {end_time - start_time}")

    many_games2(default_state, count)

         
        

def many_games2(default_state, num_many_games):        
    default_state.play_n_games_to_end(num_many_games)
    
    
    

if __name__ == "__main__":
    pytest.main()
