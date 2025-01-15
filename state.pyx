# distutils: language=c
# cython: profile=False
# cython: binding=True

# Define NPY_NO_DEPRECATED_API to silence warnings
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX, srand

# Add debug flag at module level
DEBUG = False

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

# Constants
cdef enum Player:
    WHITE = 0
    BLACK = 1
    NONE = 2

# Board positions
cdef int BAR_POS = 0
cdef int BEAR_OFF_POS = 25
cdef int HOME_START_POS = 18
cdef int OPP_BAR_POS = 25

# Optimized data structures
cdef struct Move:
    unsigned char src
    unsigned char n

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
cdef class BoardState:
    """Static methods for board operations to support efficient move generation"""
    
    @staticmethod
    cdef bint can_bear_off(unsigned char[:, ::1] board) nogil:
        cdef unsigned char pip_count = 0
        cdef int i
        
        for i in range(HOME_START_POS, BEAR_OFF_POS + 1):
            pip_count += board[0, i]
        
        return pip_count == 15
    
    @staticmethod
    cdef bint can_move_pip(unsigned char[:, ::1] board, unsigned char src, unsigned char n) nogil:
        
        
        # do we have to move from the bar
        if board[0, BAR_POS] > 0 and src != BAR_POS:
            return False
        
        #is there a piece to move? 
        if board[0, src] == 0:
            return False
            
        cdef unsigned char dst = src + n
        cdef unsigned char i

        if dst > BEAR_OFF_POS:
            if not BoardState.can_bear_off(board):
                return False
                
            for i in range(HOME_START_POS, src):
                if board[0, i] > 0:
                    return False
            dst = BEAR_OFF_POS
            
        if dst < BEAR_OFF_POS and board[1, dst] >= 2:
            return False
            
        return True

    @staticmethod
    cdef void apply_move(unsigned char[:, ::1] board, Move move) nogil:
        cdef unsigned char dst = min(move.src + move.n, BEAR_OFF_POS)

        
        
        board[0, move.src] -= 1
        board[0, dst] += 1
        
        if dst < BEAR_OFF_POS:
            if board[1, dst] == 1:
                board[1, dst] = 0
                board[1, OPP_BAR_POS] += 1
            elif board[1, dst] > 1:
                raise "Error: More than one opponent piece on destination, this should be blocked"
    
    @staticmethod
    cdef void sanity_checks(unsigned char[:, ::1] board):
        
        #sum each row and ensure it is 15
        cdef int i
        for i in range(2):
            if sum(board[i, :]) != 15:
                raise "Error: Invalid board state, sum of row is not 15"
        
        #make sure white and black are not in the same position
        for i in range(25):
            if board[0, i] > 0 and board[1, i] > 0:
                raise "Error: Invalid board state, white and black in the same position"
        

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class State:
    """Main game state class"""
    cdef:
        #unsigned char[:, ::1] full_board
        public unsigned char[:, ::1] board
        public unsigned char[:, ::1] board_white
        public unsigned char[:, ::1] board_black
        public unsigned char[:, ::1] board_curr
        public unsigned char[:, ::1] board_opp
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
        #self.board_opp = NULL # can't assign NULL to memory view
        #self.board_curr = NULL # can't assign NULL to memory view
        self.dice = [0, 0]
        self.move_seq_list = []
        

        self.set_board(np.array([
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]
        ], dtype=np.uint8))

    # Inside the State class in state.pyx
    cpdef set_dice(self, dice):
        """Set dice values for testing purposes."""
        if dice[0] > 0 and dice[0] < 7 and dice[1] > 0 and dice[1] < 7:
            self.dice[0], self.dice[1] = dice

    cpdef void set_board(self, board_array):

        if isinstance(board_array, np.ndarray):
            self.board = np.ascontiguousarray(board_array, dtype=np.uint8)
        else:
            self.board = np.ascontiguousarray(np.array(board_array, dtype=np.uint8))
        
        self.board_white = self.board[:, 0:25]  # White's view
        self.board_black = np.ascontiguousarray(self.board[::-1, ::-1])  # Black's view

        if self.player != NONE:
            # call set player ot the current player to setup boards correctly
            self.set_player(self.player)
    
    cpdef void pick_first_player(self):
        while self.player == NONE:
            DiceRollHelper.random(self.dice)
            if not self.dice[0] == self.dice[1]:
                if self.dice[0] > self.dice[1]:
                    self.set_player(WHITE)
                else:
                    self.set_player(BLACK)
    
    cpdef void set_player(self, int player):
        self.player = player
        if player == WHITE:
            self.board_curr = self.board_white
            self.board_opp = self.board_black
        else:
            self.board_curr = self.board_black
            self.board_opp = self.board_white
    
    cpdef void roll_dice(self):
        DiceRollHelper.random(self.dice)

    cdef bint is_dice_valid(self):
        if self.dice[0] > 0 and self.dice[0] < 7 and \
           self.dice[1] > 0 and self.dice[1] < 7:
           return True
        return False
    
    cpdef list get_legal_moves(self):
        if self.player == NONE:
            raise "Error player is not set yet"

        if not self.is_dice_valid():
            raise f"Error Dice not valid: dice: {self.dice}"
        
        if DEBUG:
            print(f"Player: {self.player} Valid Dice: {self.is_dice_valid()} Dice: {self.dice}")
        return MoveGenerator.generate_moves(self.board_curr, self.dice[0], self.dice[1])
    
    cpdef void do_moves(self, MoveSequence moveSeq):
        cdef int i

        assert self.player != NONE, "Player must be set before making moves"
        
        
        if moveSeq:
            self.move_seq_list.append(moveSeq.copy())
            for i in range(moveSeq.n_moves):
                if not BoardState.can_move_pip(self.board_curr ,moveSeq.moves[i].src, moveSeq.moves[i].n):
                    if DEBUG:
                        print(f"Invalid move: {moveSeq.moves[i].src} {moveSeq.moves[i].n}")
                        print(np.array_str(self.board_curr, precision=2, suppress_small=True))
                    raise f"Invalid move: {moveSeq.moves[i].src} {moveSeq.moves[i].n}"

                BoardState.apply_move(self.board_curr, moveSeq.moves[i])
                
                if DEBUG:
                    BoardState.sanity_checks(self.board_curr)

        self._goto_next_turn()
    
    cdef void _goto_next_turn(self):
        self.check_for_winner()
        if not self.isTerminal():
            self.set_player(1 - self.player)
        
    
    cpdef bint isTerminal(self):
        return self.winner != NONE
    
    cpdef void check_for_winner(self):
        if self.board_curr[0, BEAR_OFF_POS] == 15:
            self.winner = self.player
            self.points = self._calculate_winner_points()
    
    cdef int _calculate_winner_points(self):
        if self.winner == NONE:
            return 0

        if self.board_opp[0, BEAR_OFF_POS] > 0:
            return 1

        cdef int i
        for i in range(HOME_START_POS, BEAR_OFF_POS):
            if self.board_opp[1, i] > 0:
                return 3
        
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


cdef class MoveSequence:
    cdef:
        public Move[4] moves
        public unsigned char[2] dice
        public unsigned char n_moves
        public unsigned char[:, ::1] final_board
        public bint has_final_board

    def __cinit__(self, dice=None):
        self.n_moves = 0
        self.has_final_board = False  # Initialize flag
        self.dice = [0, 0]  # Initialize to zeros
        if dice is not None:
            self.dice[0] = dice[0]
            self.dice[1] = dice[1]

    cpdef list get_moves_tuple(self):
        """Return a list of move tuples (src, n)"""
        cdef list result = []
        cdef int i
        for i in range(self.n_moves):
            result.append((self.moves[i].src, self.moves[i].n))
        return result

    cdef Move* get_moves(self):
        """Return a pointer to the moves array"""
        return &self.moves[0]
    
    cpdef MoveSequence add_move(self, unsigned char src, unsigned char n):
        if self.n_moves < 4:
            self.moves[self.n_moves].src = src
            self.moves[self.n_moves].n = n
            self.n_moves += 1
        return self

    cpdef MoveSequence add_move_o(self, Move move):
        if self.n_moves < 4:
            self.moves[self.n_moves] = move
            self.n_moves += 1
        return self
    
    cdef MoveSequence copy(self):
        cdef MoveSequence new_seq = MoveSequence()
        cdef int i
        new_seq.dice[0] = self.dice[0]
        new_seq.dice[1] = self.dice[1]
        for i in range(self.n_moves):
            new_seq.add_move(self.moves[i].src, self.moves[i].n)
        
        # Copy final board state if it exists
        if self.has_final_board:
            new_seq.set_final_board(self.final_board)
       
        return new_seq
    
    cdef void set_final_board(self, unsigned char[:, ::1] board):
        cdef unsigned char[:, ::1] board_copy = np.asarray(board).copy()
        self.final_board = board_copy
        self.has_final_board = True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MoveGenerator:
    """Static methods for generating legal moves"""
    
    @staticmethod
    cdef list generate_moves(unsigned char[:, ::1] board, unsigned char d1, unsigned char d2):
        cdef list all_sequences = []
        cdef unsigned char[2] reverse_dice
        cdef MoveSequence curr_seq
    
        if DEBUG:
            print(f"Generating moves with dice {d1} {d2} :\n")
            print("Calling _gen_moves_recuriseve")

        
        #This covers doubles and first order non-double
        MoveGenerator._generate_moves_recursive(
            board,
            0,
            d1,
            d2,
            MoveSequence([d1, d2]),
            all_sequences
        )
        if DEBUG:
            print("Returned from _gen_moves_recursive")

        if d1 != d2:
            # this changes the order of 2 die
            reverse_dice[0] = d2
            reverse_dice[1] = d1

            MoveGenerator._generate_moves_recursive(
                board,
                0,
                d2,
                d1,
                MoveSequence(reverse_dice),
                all_sequences
            )
        
        return MoveGenerator._filter_moves(all_sequences)
    
    @staticmethod
    cdef void _generate_moves_recursive(
        unsigned char[:, ::1] board,
        unsigned char move_num,
        unsigned char d1,
        unsigned char d2,
        MoveSequence curr_sequence,
        list all_sequences
    ):
        cdef int src
        cdef Move move
        cdef unsigned char[:, ::1] new_board
        cdef unsigned char die_value
        cdef bint found_one_board = False
        cdef MoveSequence new_sequence
        cdef bint isDouble = curr_sequence.dice[0] == curr_sequence.dice[1]
        
        if DEBUG:
            print(f"_gen_moves Move number: {move_num}, Die value: {curr_sequence.dice}, Current moves: {curr_sequence.moves}")
            print(f"Len of all_seq: {len(all_sequences)},")

        if len(all_sequences) > 1000:
            if DEBUG:
                print(f"Max sequences hit 1000")
            return
        
        if move_num == 0:
            die_value = d1
        elif move_num == 1:
            die_value = d2
        elif not isDouble and move_num == 2:
            curr_sequence.set_final_board(board)  # Store final board state
            all_sequences.append(curr_sequence.copy()) # do we need a copy here?   this should be the final board. 
            return
        elif isDouble and move_num < 4:
            die_value = curr_sequence.dice[0]
        elif move_num == 4:
            curr_sequence.set_final_board(board)  # Store final board state
            all_sequences.append(curr_sequence.copy()) # do we need a copy here?   this should be the final board. 
            return
        else:
            # Debug print statement
            if DEBUG:
                print(f"Move number: {move_num}, Die value: {die_value}, Current sequence: {curr_sequence}, Board state:\n{board}")
            
            #assert False, "Invalid move_num in recursive generation"
        
        if DEBUG:
            print(f"_gen_moves Die value: {die_value}, Current moves: {curr_sequence.moves}")

        
        for src in range(BAR_POS, BEAR_OFF_POS): #BEAROFF is not included
            if BoardState.can_move_pip(board, src, die_value):
                move.src = src
                move.n = die_value
                
                new_board = np.asarray(board).copy()
                BoardState.apply_move(new_board, move)

                new_sequence = curr_sequence.copy()
                new_sequence.add_move(move.src, move.n)
                
                MoveGenerator._generate_moves_recursive(
                    new_board,
                    move_num + 1,
                    d1,
                    d2,
                    new_sequence,
                    all_sequences
                )
                found_one_board = True
        
        if not found_one_board:
            # didn't find any other possible moves this may be the end of the sequence we need to save this one
            # otherwise if more were found then this is an invalid move by definition

            curr_sequence.set_final_board(board)  # Store final board state
            all_sequences.append(curr_sequence.copy()) # again is a copy necessary? 
            return

    
    @staticmethod
    cdef bytes _board_to_bytes(unsigned char[:, ::1] board) :
        # Convert board state to a bytes representation for hashing
        return bytes(board.tobytes())

    @staticmethod
    cdef list _filter_moves(list sequences):
        if not sequences:
            return []
            
        # First filter by maximum moves used
        cdef int max_die
        cdef int max_moves = max(seq.n_moves for seq in sequences)
        cdef set unique_states = set()
        cdef list unique_sequences = []
        cdef list sequences2 = []
        cdef bytes board_hash
        cdef MoveSequence seq
        
        #filters sequences with less than max moves
        sequences = [seq for seq in sequences if seq.n_moves == max_moves]
        
        #filter sequences with less than max die if only 1 move possible.
        if max_moves == 1:
            max_die = 0
            for seq in sequences:
                if seq.moves[0].n > max_die:
                    max_die = seq.moves[0].n
            
            for seq in sequences:
                if seq.moves[0].n == max_die:
                    sequences2.append(seq)
        
        # Filter duplicates using the stored final board states
        
        
        for seq in sequences:
            if not seq.has_final_board:
                assert False, "Sequence does not have final board state"
            
            # Get hashable representation of stored final board state
            board_arr = np.asarray(seq.final_board)
            board_hash = board_arr.tobytes()
            
            # Only keep sequence if board state is unique
            if board_hash not in unique_states:
                unique_states.add(board_hash)
                unique_sequences.append(seq)
        
        return unique_sequences