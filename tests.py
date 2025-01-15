import numpy as np
import unittest
from state import State, set_debug, set_random_seed, MoveSequence

# Constants for player colors
WHITE = 0
BLACK = 1
NONE = 2



class TestBackgammonState(unittest.TestCase):
    def setUp(self):
        set_debug(True)  # Enable debugging for detailed move generation info
        set_random_seed(42)  # Use consistent seed for reproducibility
    
    def test_initial_board_state(self):
        """Test that the initial board state is set up correctly"""
        state = State()
        
        # Get board through move generation to verify public interface
        state.pick_first_player()

        moves = state.get_legal_moves()
        self.assertTrue(len(moves) > 0, "Should generate initial moves")
        
        # Check initial positions match what we expect
        board = state.board
        self.assertEqual(board[0, 1], 2, "White should have 2 pieces on point 1")
        self.assertEqual(board[0, 12], 5, "White should have 5 pieces on point 12")
        self.assertEqual(board[0, 19], 5, "White should have 5 pieces on point 19")
        
        self.assertEqual(board[1, 5], 5, "Black should have 5 pieces on point 5")
        self.assertEqual(board[1, 12], 5, "Black should have 5 pieces on point 12")
        self.assertEqual(board[1, 23], 2, "Black should have 2 pieces on point 23")

    def test_move_sequence(self):
        """Test MoveSequence operations including copying"""
        import numpy as np
        from state import MoveSequence, State
        
        state = State()
        
        # Create a simple board with one piece
        board_array = np.zeros((2, 25), dtype=np.uint8)
        board_array[0, 1] = 1
        state.set_board(board_array)
        
        # Create and populate a move sequence
        seq = MoveSequence()
        seq.add_move(1, 6)  # Move from point 1 by 6 spaces
        
        # Test n_moves accessibility
        self.assertEqual(seq.n_moves, 1, "Move sequence should have 1 move")
        

    
    def test_can_bear_off(self):
        """Test the bearing off rules through move generation"""
        state = State()
        
        # Setup position where White can bear off
        board_array = np.zeros((2, 25), dtype=np.uint8)
        for i in range(18, 24):  # Put all pieces in home board
            board_array[0, i] = 2
        board_array[0, 24] = 3  # Already beared off 3 pieces
        state.set_board(board_array)
        state.set_player(WHITE)
        
        # Roll a 6,5 and check if bearing off moves are generated
        state.dice = [6, 5]
        moves = state.get_legal_moves()
        
        has_bearing_off_move = False
        for move_seq in moves:
            if move_seq.moves[0]['n'] == 6 and move_seq.moves[0]['src'] >= 18:
                has_bearing_off_move = True
                break
                
        self.assertTrue(has_bearing_off_move, "Should generate bearing off moves when legal")
        
        # Move a piece out of home board and verify can't bear off
        board_array[0, 17] = 1
        board_array[0, 18] = 1
        state.set_board(board_array)
        
        moves = state.get_legal_moves()
        has_bearing_off_move = False
        for move_seq in moves:
            if move_seq.moves[0]['src'] >= 18 and move_seq.moves[0]['n'] > (24 - move_seq.moves[0]['src']):
                has_bearing_off_move = True
                break
                
        self.assertFalse(has_bearing_off_move, 
                        "Should not generate bearing off moves when pieces are outside home")
    
    def test_dice_roll_generation(self):
        """Test dice roll generation and handling"""
        set_random_seed(42)  # Ensure reproducible rolls
        state = State()
        
        # Test initial roll
        state.pick_first_player()
        roll_sum = state.dice[0] + state.dice[1]
        self.assertTrue(2 <= roll_sum <= 12, "Initial roll should be valid")
        
        # Test doubles handling

        state.dice = [6, 6]
 
        moves = state.get_legal_moves()
        self.assertTrue(len(moves) > 0, "Should generate moves for doubles")
        
        # Check some moves use all four dice parts
        has_four_part_move = False
        for move_seq in moves:
            if move_seq.n_moves == 4:
                has_four_part_move = True
                break
        self.assertTrue(has_four_part_move, "Should generate four-part moves for doubles")
        
        # Test non-doubles
        state.dice = [6, 3]
  
        moves = state.get_legal_moves()
        self.assertTrue(len(moves) > 0, "Should generate moves for non-doubles")
    
    def test_forced_bar_moves(self):
        """Test that pieces must come off the bar first"""
        state = State()
        
        # Setup position with pieces on bar
        board_array = np.zeros((2, 25), dtype=np.uint8)
        board_array[0, 0] = 2  # Two pieces on bar
        board_array[0, 5] = 5  # Some other pieces
        board_array[0, 7] = 8
        state.set_board(board_array)
        
        state.dice = [6, 3]
        state.set_player(WHITE)
        moves = state.get_legal_moves()
        
        # All moves should start from bar
        for move_seq in moves:
            self.assertEqual(move_seq.moves[0]["src"], 0, 
                           "First move must be from bar when pieces are on bar")
            
    def test_move_execution(self):
        """Test that moves execute correctly"""
        state = State()
        
        # Setup a simple position
        board_array = np.zeros((2, 25), dtype=np.uint8)
        board_array[0, 1] = 1  # One piece that can move
        state.set_board(board_array)
        
        # Roll a 6
        state.dice = [6, 1]
        state.set_player(WHITE)
        # Get and execute a move
        moves = state.get_legal_moves()
        self.assertTrue(len(moves) > 0, "Should generate at least one move")
        
        initial_board = state.board.copy()
        state.do_moves(moves[0])
        
        # Verify the piece moved
        self.assertEqual(state.board[0, 1], initial_board[0, 1] - 1,
                        "Source point should have one less piece")
        self.assertEqual(state.board[0, 7], initial_board[0, 7] + 1,
                        "Destination point should have one more piece")


    def test_blot_hitting(self):
        """Test hitting single pieces (blots)"""
        state = State()        
        # White to play and can hit
        state.set_dice([6, 3])
        state.set_player(WHITE)
       
        moves = MoveSequence()
        moves.add_move(1,1).add_move(1,2)

        state.do_moves(moves)
        state.set_dice([4, 5])

        moves = MoveSequence()
        moves.add_move(18,4).add_move(18,5)
        state.do_moves(moves)

        state.set_dice([5, 1])
        moves = state.get_legal_moves()
    
        self.assertEqual(state.board[1, 25], 2, "Hit piece should go to bar")

if __name__ == '__main__':
    unittest.main()