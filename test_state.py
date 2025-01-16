import pytest
import numpy as np
from bg_game import set_debug, get_debug, set_random_seed, BGGame

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
    
    set_random_seed(42)
    state.roll_dice()
    d1 = state.dice
    
    set_random_seed(42)
    state.roll_dice()
    d2 = state.dice

    assert d1 == d2

# State Initialization Tests
def test_state_initialization(default_state):
    """Verify that the State object initializes correctly."""
    assert default_state is not None
    assert isinstance(default_state, State)

# Dice Roll Operations
def test_dice_roll():
    """Verify dice roll logic."""
    state = BGGame()
    state.roll_dice()
    dice = b'\x02\x05'
    state.set_dice(dice)
    assert state.dice == b'\x02\x05'

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
    assert isinstance(valid_moves, list)
    for moveSeq in valid_moves:
        for i in range(moveSeq.n_moves):
            assert moveSeq.moves[i]['src'] >= 0
            assert moveSeq.moves[i]['src'] < 25
            assert moveSeq.moves[i]['n'] > 0
            assert moveSeq.moves[i]['n'] < 7

# Edge Case Tests
def test_no_valid_moves(default_state):
    """Test a board state where no moves are possible."""
    set_debug(True)
    default_state.set_player(WHITE)
    default_state.board[0, :] = 0  # Empty white's board
    default_state.board[0, 0] = 15  # Place all checkers on the bar

    default_state.set_dice((6,6))
    valid_moves = default_state.get_legal_moves()

    assert valid_moves == []

# Boundary Conditions
def test_bar_and_bear_off_logic(default_state):
    """Test moves from bar and bear-off positions."""

    default_state.board[0, 0] = 2  # Place checkers on bar
    default_state.board[0, 1] = 0  # Place checkers on bear-off
    default_state.set_player(WHITE)
    default_state.set_dice((3, 4))

    valid_moves = default_state.get_legal_moves()
    assert len(valid_moves) == 1
    moveSeq = valid_moves[0]
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
    set_random_seed(42)
    default_state.set_player(WHITE)
    default_state.set_dice((6, 1))

    moves = default_state.get_legal_moves()
    set_random_seed(42)
    repeated_moves = default_state.get_legal_moves()

    assert moves == repeated_moves

def test_seeded_move_generation(default_state):
    """Capture and validate move generation with seeded randomness."""
    set_random_seed(42)
    default_state.set_player(WHITE)
    default_state.set_dice((6, 1))

    moves = default_state.get_legal_moves()
    set_random_seed(42)
    repeated_moves = default_state.get_legal_moves()

    assert moves == repeated_moves

def test_many_moves(default_state):
    """Capture and validate move generation with seeded randomness."""

    default_state.set_player(WHITE)
    default_state.set_dice((3, 3))
    board = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 3, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 5]
    ])

    default_state.set_board(board)
    moves = default_state.get_legal_moves()
    assert len(moves) == 368

    default_state.reset()
    default_state.set_player(WHITE)
    default_state.set_dice((1, 1))
    board = np.array([
        [0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 5]
    ])

    default_state.set_board(board)
    moves = default_state.get_legal_moves()
    assert len(moves) == 1107

    default_state.reset()
    default_state.set_player(WHITE)
    default_state.set_dice((2, 1))
    board = np.array([
        [0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 5]
    ])

    default_state.set_board(board)
    moves = default_state.get_legal_moves()
    assert len(moves) == 161

def test_many_games(default_state):
    """Capture and validate move generation with seeded randomness."""
    count = 1000
    print("Starting games {count} each . is 100")
    for x in range(count):
        default_state.reset()
        default_state.set_player(WHITE)
    
        while not default_state.isTerminal():
            default_state.roll_dice()
            moves = default_state.get_legal_moves()
            if len(moves) > 0:
                move = moves[np.random.randint(len(moves))]
                default_state.do_moves(move)
            else:
                default_state.do_moves(None)
        
            
            
    
        
    
    

if __name__ == "__main__":
    pytest.main()
