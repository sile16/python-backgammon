# distutils: language=c
# cython: profile=False
# cython: binding=True

# Define NPY_NO_DEPRECATED_API to silence warnings
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX, srand

from bg_common cimport *
from bg_board cimport BoardState
from bg_moves cimport MoveSequence, MoveGenerator



# Export these functions at module level
__all__ = ['set_debug', 'get_debug', 'set_random_seed', 'State']

# Add global function to set debug mode
def set_debug(enabled):
    """Python-accessible wrapper for debug mode setting"""
    global DEBUG
    DEBUG = enabled

def get_debug():
    """Python-accessible wrapper for getting debug mode"""
    global DEBUG
    return DEBUG

# Add function to set random seed
def set_random_seed(seed):
    """Python-accessible wrapper for setting random seed"""
    srand(seed)





@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DiceRollHelper:
    """Helper class for DiceRoll operations"""
    
    @staticmethod
    cdef void create(unsigned char[2] dice, unsigned char d1, unsigned char d2) nogil:
        dice[0] = d1
        dice[1] = d2
    
    @staticmethod
    cdef void random(unsigned char[2] dice) nogil:
        dice[0] = 1 + (rand() % 6)
        dice[1] = 1 + (rand() % 6)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class BGGame:
    """Main game state class"""
    cdef:
        public np.ndarray board
        public np.ndarray board_curr
        public int player
        public int winner
        public unsigned char[2] dice
        public list legal_moves
        public int points
        public list move_seq_list
    
    def __cinit__(self):
        self.reset()
    
    cpdef int get_move_count(self):
        return len(self.move_seq_list)
    
    cpdef void reset(self):
        self.player = NONE
        self.winner = NONE
        self.legal_moves = []
        self.points = 0
        self.dice[0] = 0 # can't assign with [0, 0] as it changes the types and breaks the pointer type
        self.dice[1] = 0
        self.move_seq_list = []
        
        
        # Initialize board with starting position
        self.board = np.zeros((2, 26), dtype=np.uint8)
        # Set starting positions for both players
        self.set_board([
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]
        ])

    cpdef void set_board(self, board_array):
        if isinstance(board_array, np.ndarray):
            self.board = np.array(board_array, dtype=np.uint8)
        else:
            self.board = np.array(board_array, dtype=np.uint8)
        
        self.board_curr = self.board[:,:]
    
    cpdef np.ndarray get_board(self):
        return np.array([
            self.board[WHITE],
            self.board[BLACK]  # Reverse black's view
        ], dtype=np.uint8)

    cpdef np.ndarray get_board_white_black(self):
        return np.array([
            self.board[WHITE],
            self.board[BLACK][::-1]
        ], dtype=np.uint8)
    
    cpdef bint can_bear_off(self):
        return BoardState.can_bear_off(self.board_curr)
    
    cpdef void set_player(self, int player):
        if DEBUG:
            print()
            print(f"Before player = {self.player}")
            print("[[0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5]]")
            print(np.array_str(self.board_curr, precision=2, suppress_small=True))
            BoardState.sanity_checks(self.board_curr)

        self.player = player
        if player == WHITE:
            self.board_curr = self.board[:,:]
        else:
            self.board_curr = self.board[::-1, ::-1]

        if DEBUG:
            print(f"After Set Player {self.player}")
            print("[[0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5]]")
            print(np.array_str(self.board_curr, precision=2, suppress_small=True))
            BoardState.sanity_checks(self.board)
    
    cpdef set_dice(self, dice):
        """Set dice values for testing purposes."""
        if dice[0] > 0 and dice[0] < 7 and dice[1] > 0 and dice[1] < 7:
            self.dice[0], self.dice[1] = dice
    
    cpdef void pick_first_player(self):
        while self.player == NONE:
            DiceRollHelper.random(self.dice)
            if not self.dice[0] == self.dice[1]:
                if self.dice[0] > self.dice[1]:
                    self.set_player(WHITE)
                else:
                    self.set_player(BLACK)
    
    cpdef void roll_dice(self):
        DiceRollHelper.random(self.dice)

    cdef bint is_dice_valid(self):
        if self.dice[0] > 0 and self.dice[0] < 7 and \
           self.dice[1] > 0 and self.dice[1] < 7:
           return True
        return False
    
    cpdef list get_legal_moves(self):
        if self.player == NONE:
            raise ValueError("Error player is not set yet")

        if not self.is_dice_valid():
            raise ValueError(f"Error Dice not valid: dice: {self.dice}")
        
        if DEBUG:
            # Print dice in decimal format
            print(f"Player: {self.player} Valid Dice: {self.is_dice_valid()} Dice: [{self.dice[0]}, {self.dice[1]}]")
            
        return MoveGenerator.generate_moves(self.board_curr, self.player, self.dice[0], self.dice[1])
    
    cpdef void do_moves(self, MoveSequence moveSeq):
        cdef int i

        assert self.player != NONE, "Player must be set before making moves"
        
        if moveSeq:
            self.move_seq_list.append(moveSeq.copy())
            for i in range(moveSeq.n_moves):
                
                if DEBUG:
                    BoardState.sanity_checks(self.board)
                    if not BoardState.can_move_pip(self.board_curr, moveSeq.moves[i].src, moveSeq.moves[i].n):
                        print(f"Invalid move: {moveSeq.moves[i].src} {moveSeq.moves[i].n}")
                        raise ValueError(f"Invalid move: {moveSeq.moves[i].src} {moveSeq.moves[i].n}" )        
            
                BoardState.apply_move(self.board_curr, moveSeq.moves[i])


        self._goto_next_turn()
    
    cdef void _goto_next_turn(self):
        self._check_for_winner()
        if not self.isTerminal():
            self.set_player(1 - self.player)
    
    cpdef bint isTerminal(self):
        return self.winner != NONE
    
    cdef void _check_for_winner(self):
        if self.board_curr[0, BEAR_OFF_POS] == 15:
            self.winner = self.player
            self.points = self._calculate_winner_points()
    
    cdef int _calculate_winner_points(self):
        if self.winner == NONE:
            return 0

        #opponent has born off at least 1
        if self.board_curr[1, OPP_BEAR_OFF_POS] > 0:
            return 1

        # check for opponent pieces in curr home / opponent BAR
        cdef int i
        for i in range(HOME_START_POS, OPP_BAR_POS):
            if self.board_curr[1, i] > 0:
                return 3 # backgammon
        
        return 2  # Gammon
    
    cpdef int play_game_to_end(self):
        cdef int i, n_moves
        cdef list moves
        if self.player == NONE:
            self.pick_first_player()
            self.roll_dice()
        while not self.isTerminal():
            moves = self.get_legal_moves()
            n_moves = len(moves)
            if n_moves > 0:
                i = rand() % n_moves
                self.do_moves(moves[i])
            self.roll_dice()
        return self.winner

