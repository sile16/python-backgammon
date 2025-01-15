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



cdef class BoardState:
    """Static methods for board operations to support efficient move generation"""
    
    @staticmethod
    cdef bint can_bear_off(np.ndarray[np.uint8_t, ndim=2] board):
        cdef unsigned char pip_count = 0
        cdef int i
        
        for i in range(HOME_START_POS, BEAR_OFF_POS + 1):
            pip_count += board[0, i]
        
        return pip_count == 15
    
    @staticmethod
    cdef bint can_move_pip(np.ndarray[np.uint8_t, ndim=2] board, unsigned char src, unsigned char n) :
        cdef unsigned char dst
        cdef unsigned char i

        # Must move from bar first
        if board[0, BAR_POS] > 0 and src != BAR_POS:
            return False
        
        # Must have a piece to move
        if board[0, src] == 0:
            return False
            
        dst = src + n
        
        # Handle bearing off
        if dst >= BEAR_OFF_POS:
            if not BoardState.can_bear_off(board):
                return False
            
            if dst > BEAR_OFF_POS:
                for i in range(HOME_START_POS, src):
                    if board[0, i] > 0:
                        return False

            return True
            
        # Check if destination is blocked by opponent
        if board[1, dst] > 1:
            return False
            
        return True

    @staticmethod
    cdef void apply_move(np.ndarray[np.uint8_t, ndim=2] board, Move move):
        cdef unsigned char dst = min(move.src + move.n, BEAR_OFF_POS)

        if DEBUG:
            print(f"Applying move: {move.src} {move.n}")
            print(f"Board state:\n{board}")
            print("[[0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5]]")
            print(np.array_str(board, precision=2, suppress_small=True))
        
        # Move piece from source to destination
        board[0, move.src] -= 1
        board[0, dst] += 1
        
        # Handle hitting opponent's blot
        if dst < BEAR_OFF_POS:
            
            if board[1, dst] == 1:
                board[1, dst] = 0
                board[1, OPP_BAR_POS] += 1
            elif board[1, dst] > 1:
                raise ValueError("Error: More than one opponent piece on destination")

        if DEBUG:
            print(f"done with move: {move.src} {move.n}")
            print(f"Board state:\n{board}")
            print("[[0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5]]")
            print(np.array_str(board, precision=2, suppress_small=True))
            print("")
            print("")

    @staticmethod
    cdef void sanity_checks(np.ndarray[np.uint8_t, ndim=2] board):
        cdef int i
        cdef int sum_white = 0
        cdef int sum_black = 0
        
        # Check piece counts
        for i in range(26):
            sum_white += board[0, i]
            sum_black += board[1, i]
        
        if sum_white != 15 or sum_black != 15:
            #print sums:
            print(f"sum_white: {sum_white} sum_black: {sum_black}")
            print(f"Board state:\n{board}")
            print("[[0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5]]")
            print(np.array_str(board, precision=2, suppress_small=True))
            raise ValueError("Error: Invalid board state, piece count not 15")
        
        # Check for overlapping pieces
        for i in range(1, 25):
            if board[0, i] > 0 and board[1, i] > 0:
                print(f"Board state:\n{board}")
                print("[[0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5]]")
                print(np.array_str(board, precision=2, suppress_small=True))
                raise ValueError("Error: Invalid board state, overlapping pieces")

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class State:
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
        self.dice = [0, 0]
        self.move_seq_list = []
        
        
        # Initialize board with starting position
        self.board = np.zeros((2, 26), dtype=np.uint8)
        # Set starting positions for both players
        self.set_board([
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]
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
    
    cpdef void set_player(self, int player):
        

        if True:
            print("Set Player Before")
            print()
            print("[[0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5]]")
            print(np.array_str(self.board_curr, precision=2, suppress_small=True))
            print("player = {self.player}")
            BoardState.sanity_checks(self.board_curr)

        self.player = player
        if player == WHITE:
            self.board_curr = self.board[:,:]
        else:
            self.board_curr = self.board[::-1, ::-1]

        if True:
            print(f"After Set {player}")
            print(f"Board state:\n{self.board}")
            print("player = {self.player}")
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
        self.check_for_winner()
        if not self.isTerminal():
            self.set_player(1 - self.player)
    
    cpdef bint isTerminal(self):
        return self.winner != NONE
    
    cpdef void check_for_winner(self):
        if self.board_curr[self.player, BEAR_OFF_POS] == 15:
            self.winner = self.player
            self.points = self._calculate_winner_points()
    
    cdef int _calculate_winner_points(self):
        if self.winner == NONE:
            return 0

        
        if self.board_curr[1, BEAR_OFF_POS] > 0:
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

cdef class MoveSequence:
    cdef:
        public Move[4] moves
        public unsigned char[2] dice
        public unsigned char n_moves
        public np.ndarray final_board
        public bint has_final_board

    def __cinit__(self, dice=None):
        self.n_moves = 0
        self.has_final_board = False
        self.dice = [0, 0]
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
        
        if self.has_final_board:
            new_seq.set_final_board(self.final_board)
       
        return new_seq
    
    cdef void set_final_board(self, np.ndarray[np.uint8_t, ndim=2] board):  
        self.final_board = board.copy()
        self.has_final_board = True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MoveGenerator:
    """Static methods for generating legal moves"""
    
    @staticmethod
    cdef list generate_moves(np.ndarray[np.uint8_t, ndim=2] board, int player, unsigned char d1, unsigned char d2):
        cdef list all_sequences = []
        cdef unsigned char[2] reverse_dice
        cdef MoveSequence curr_seq
    
        if DEBUG:
            print(f"Generating moves for player {player} with dice {d1} {d2}")

        # Create initial MoveSequence with dice
        curr_seq = MoveSequence()
        curr_seq.dice[0] = d1
        curr_seq.dice[1] = d2
        
        # Generate moves with dice in original order
        MoveGenerator._generate_moves_recursive(
            board,
            player,
            0,
            d1,
            d2,
            curr_seq,
            all_sequences
        )

        # For non-doubles, try reverse dice order
        if d1 != d2:
            curr_seq = MoveSequence()
            curr_seq.dice[0] = d2
            curr_seq.dice[1] = d1

            MoveGenerator._generate_moves_recursive(
                board,
                player,
                0,
                d2,
                d1,
                curr_seq,
                all_sequences
            )
        
        return MoveGenerator._filter_moves(all_sequences)
    
    @staticmethod
    cdef void _generate_moves_recursive(
        np.ndarray[np.uint8_t, ndim=2] board,
        int player,
        unsigned char move_num,
        unsigned char d1,
        unsigned char d2,
        MoveSequence curr_sequence,
        list all_sequences
    ):
        cdef int src
        cdef Move move
        cdef np.ndarray[np.uint8_t, ndim=2] new_board
        cdef unsigned char die_value
        cdef bint found_valid_move = False
        cdef MoveSequence new_sequence
        cdef bint isDouble = curr_sequence.dice[0] == curr_sequence.dice[1]

        move.src = 0
        move.n = 0
        
        if DEBUG:
            print(f"_gen_moves Move number: {move_num}, Dice: {curr_sequence.dice[0]} {curr_sequence.dice[1]}")

        # Limit recursion depth for safety
        if len(all_sequences) > 1000:
            if DEBUG:
                print("Max sequences limit reached (1000)")
            return
        
        # Determine which die to use for this move
        if move_num == 0:
            die_value = d1
        elif move_num == 1:
            die_value = d2
        elif not isDouble and move_num == 2:
            # For non-doubles, end after two moves
            curr_sequence.set_final_board(board)
            all_sequences.append(curr_sequence.copy())
            return
        elif isDouble and move_num < 4:
            # For doubles, allow up to four moves
            die_value = curr_sequence.dice[0]
        elif move_num == 4:
            # End after four moves for doubles
            curr_sequence.set_final_board(board)
            all_sequences.append(curr_sequence.copy())
            return
        else:
            if DEBUG:
                print(f"Invalid move_num {move_num} in recursive generation")
            return
        
        # Try all possible source positions
        for src in range(BAR_POS, BEAR_OFF_POS):
            if BoardState.can_move_pip(board, src, die_value):
                move.src = src
                move.n = die_value
                
                # Create new board state and apply move
                new_board = board.copy()
                
                if DEBUG:
                    print(f"new board Applying move: {move.src} {move.n}")
                BoardState.apply_move(new_board, move)

                # Create new sequence with this move
                new_sequence = curr_sequence.copy()
                new_sequence.add_move(move.src, move.n)
                
                # Recurse to find subsequent moves
                MoveGenerator._generate_moves_recursive(
                    new_board,
                    player,
                    move_num + 1,
                    d1,
                    d2,
                    new_sequence,
                    all_sequences
                )
                found_valid_move = True
        
        # If no valid moves found, this might be the end of a sequence
        if not found_valid_move:
            curr_sequence.set_final_board(board)
            all_sequences.append(curr_sequence.copy())
            return
    
    @staticmethod
    cdef bytes _board_to_bytes(np.ndarray[np.uint8_t, ndim=2] board): 
        """Convert board state to bytes for hashing"""
        return board.tobytes()

    @staticmethod
    cdef list _filter_moves(list sequences):
        """Filter move sequences to remove duplicates and ensure maximum move usage"""
        if not sequences:
            return []
            
        # Find maximum number of moves used
        cdef int max_die
        cdef int max_moves = max(seq.n_moves for seq in sequences)
        cdef set unique_states = set()
        cdef list unique_sequences = []
        cdef list filtered_sequences = []
        cdef bytes board_hash
        cdef MoveSequence seq
        
        # Keep only sequences with maximum number of moves
        max_die = 0

        filtered_sequences = []
        for seq in sequences:
            if seq.n_moves == max_moves:
                filtered_sequences.append(seq)
                if max_moves == 1:
                    if seq.moves[0].n > max_die:
                        max_die = seq.moves[0].n
        sequences = filtered_sequences
        # For single moves, prefer higher die values
        if max_moves == 1:
            
            filtered_sequences = []
            for seq in sequences:
                if seq.moves[0].n == max_die:
                    filtered_sequences.append(seq)

            sequences = filtered_sequences
        
        # Filter duplicates based on final board state
        for seq in sequences:
            if not seq.has_final_board:
                raise ValueError("Sequence missing final board state")
            
            board_hash = MoveGenerator._board_to_bytes(seq.final_board)
            
            if board_hash not in unique_states:
                unique_states.add(board_hash)
                unique_sequences.append(seq)
        
        return unique_sequences