# distutils: language=c
# cython: profile=False
# cython: binding=True

# Define NPY_NO_DEPRECATED_API to silence warnings
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
cimport cython
#rand needs to be imported from libc.stdlib, also to seed and set seed
from libc.stdlib cimport rand, RAND_MAX, srand

from bg_common cimport *
from bg_board cimport BoardState
from bg_moves cimport MoveSequence, MoveGenerator
import os



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

        #use a memoryview to access the board
        public unsigned char[:, :] blots_curr # 1 for opponent blot, -1 for own blot, 0 for empty
        public unsigned char[:, :] blocks_curr # 1 for own block, -1 for opponent block, 0 for empty
        
        public bint[2] bear_off
        public int player
        public int winner
        public unsigned char[2] dice
        public unsigned char[4] remaining_dice
        public unsigned char n_legal_remaining_dice # could be less than reamining based on 
        public list legal_moves
        public int points
        public list move_seq_list
        
    
    def __cinit__(self):
        self.reset()

    def randomize_seed(self):
        srand(int.from_bytes(os.urandom(4), 'little'))  # 32-bit random seed

    cpdef BGGame copy(self):
        cdef BGGame new_game = BGGame()
        
        new_game.set_board(self.board)
        new_game.player = self.player
        new_game.winner = self.winner
        new_game.dice[0] = self.dice[0]
        new_game.dice[1] = self.dice[1]
        new_game.bear_off[0] = self.bear_off[0]
        new_game.bear_off[0] = self.bear_off[1]
        new_game.legal_moves = self.legal_moves.copy()
        new_game.points = self.points
        new_game.move_seq_list = self.move_seq_list.copy()
        new_game.n_legal_remaining_dice = self.n_legal_remaining_dice
        new_game.remaining_dice[0] = self.remaining_dice[0]
        new_game.remaining_dice[1] = self.remaining_dice[1]
        new_game.remaining_dice[2] = self.remaining_dice[2]
        new_game.remaining_dice[3] = self.remaining_dice[3]
        return new_game


    cpdef list legal_actions(self, int depth=0):
        cdef set moves_as_int = set()

        if self.isTerminal():
            return []
        """ return legal actions, if no actions available switch player and try again until legal actions found"""
        if depth > 10:
            raise ValueError("Max legal move recursion depth found")

        if len(self.legal_actions) == 0:
            self.legal_moves = MoveGenerator.generate_moves3(self.board_curr, self.dice[0], self.dice[1])
        
        # No legal moves so change player and roll_dice
        if len(self.legal_moves) == 0:
            self.set_player(1 - self.player)
            self.roll_dice()
            self.update_state()
            return self.legal_actions(depth + 1)
        
        #because you always have to play max moves, all moves sequences should be the same lenght
        #so we should look at the first one
        self.n_legal_remaining_dice = self.legal_moves[0].n_moves

        if self.dice_remaining == 1:
            self.last_move = 1
        else:
            self.last_move = 0

        for mSeq in self.legal_moves:
            num = mSeq.moves[0].src - 1
            if mSeq.moves[0].n == self.dice_remaining[0]:
                pass
            elif mSeq.moves[0].n == self.dice_remaining[1]:
                num += 24
            else:
                raise ValueError(f"Invalid move 777: {mSeq.moves[0].src} {mSeq.moves[0].n} dice: {self.dice_remaining}")

            moves_as_int.add(num)

        return list(moves_as_int)

    cdef void _no_moves_left(self, MoveSequence mseq = None):
        self.check_for_winner()
        if not self.isTerminal():
            self.set_player(1 - self.player)
            self.roll_dice()
        self.legal_moves = []

    cdef void filter_move_from_legal_actions(self, Move move):
        cdef list remaining_legal_moves = []
        cdef int i
        cdef int remaining_moves = 0

        for mSeq in self.legal_actions:
            if mSeq.moves[0].src == move.src and mSeq.moves[0].n == move.n:

                remaining_moves = mSeq.use_move(move.src, move.n)
                
                if remaining_moves == 0:
                    # means we have exhausted all moves in the sequence
                    # should be the end of this players turn
                    self._no_moves_left(mSeq)
                    self.move_seq_list.append(mSeq)
                    return

                remaining_legal_moves.append(mSeq )

        self.legal_moves = remaining_legal_moves

    cpdef tuple step(self, int action):
        """
        action 0-23 is move src position 1-24 dice0 moves
        action 24-47 is move src position 1-24 dice1 moves
        applies move and then changes turn and rolls dice if turn is complete
        """
        cdef Move m
        cdef int dice_using = 0

        if action < 24:
            m.src = action + 1
            m.n = self.dice_remaining[0]
        else:
            dice_using = 1
            m.src = action - 23
            m.n = self.dice_remaining[1]

        if m.n == 0:
            raise ValueError(f"Invalid action in step(): {action} dice: {self.remaining_dice} m: {m}")

        BoardState.apply_move(self.board_curr, m)

        self.update_state()
        
        # move dice remaining down
        for i in range(dice_using, len(self.dice_remaining) - 1):
            self.dice_remaining[i] = self.dice_remaining[i + 1]
            self.dice_remaining[i + 1] = 0
        self.n_legal_remaining_dice -= 1
        
        if self.dice_remaining[0] == 0 or self.n_legal_remaining_dice == 0:
            self._no_moves_left()
        else:
            self.filter_move_from_legal_actions(m)
        
        return ( self.get_observation(), self.points, self.isTerminal() )

    
    cpdef np.ndarray get_observation(self):
        """Returns a NumPy array representing the full game state."""
        cdef np.ndarray board_state = np.zeros(28, dtype=np.int8)  # Corrected array initialization

        # Fill board representation
        cdef int i
        for i in range(24):  # Now correctly iterating over 0-23
            board_state[i] = self.board_curr[0, i + 1] - self.board_curr[1, i + 1]  # Player's checkers are positive, opponent's negative

        # Bar and bear-off positions
        board_state[24] = self.board_curr[0, BAR_POS]   # Player's checkers on the bar
        board_state[25] = self.board_curr[0, BEAR_OFF_POS]  # Player's checkers borne off
        board_state[26] = self.board_curr[1, OPP_BAR_POS]   # Opponent's checkers on the bar
        board_state[27] = self.board_curr[1, OPP_BEAR_OFF_POS]  # Opponent's checkers borne off

        # Convert additional state elements to NumPy arrays
        cdef np.ndarray blots = np.array(self.board_blots, dtype=np.int8)  # Assuming this is a list
        cdef np.ndarray blocks = np.array(self.board_blocks, dtype=np.int8)  # Assuming this is a list
        cdef np.ndarray dice_info = np.array(self.remaining_dice, dtype=np.int8)  # Assuming dice_remaining is iterable

        # Additional flags (turn status, legal moves, etc.)
        cdef np.ndarray additional_flags = np.array([
            int(self.can_bear_off),  # Ensure this is a scalar int
            int(self.n_legal_remaining_dice)  # Ensure this is a scalar int
        ], dtype=np.int8)

        # Concatenate all elements into a single observation array
        return np.concatenate([board_state, additional_flags, dice_info, blots, blocks])

    
    cpdef int get_move_count(self):
        return len(self.move_seq_list)
    
    cpdef void reset(self):
        self.player = NONE
        self.winner = NONE
        self.legal_moves = []
        self.points = 0
        self.dice[0] = 0 # can't assign with [0, 0] as it changes the types and breaks the pointer type
        self.dice[1] = 0
        self.remaining_dice[0] = 0
        self.remaining_dice[1] = 0
        self.remaining_dice[2] = 0
        self.remaining_dice[3] = 0
        self.n_legal_remaining_dice = 0

        self.bear_off[0] = False
        self.bear_off[1] = False
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
            self.board = board_array.copy()
        else:
            self.board = np.array(board_array, dtype=np.uint8)
        
        self.set_player(self.player)
        self.update_state()
    
    cpdef np.ndarray get_board(self):
        """return board from current perspective"""
        return self.get_board_curr()

    cpdef np.ndarray get_board_curr(self):
        """Return board from current user's perspective"""
        # make a copy because board_curr changes
        return np.array(self.board_curr, dtype=np.uint8) 

    cpdef np.ndarray get_board_white_black(self):
        return self.board

    cpdef np.ndarray get_board_black_white(self):
        return np.array(self.board[::-1, ::-1], dtype=np.uint8)

    cpdef bint can_bear_off(self):
        return BoardState.can_bear_off(self.board_curr)

    cpdef void update_state(self):
        #if self.board_curr == None:
        return

        for i in range(24):
            #blots and blocks
            if self.board_curr[0, i] == 1:
                self.board_blots[i] = 1
                self.board_blocks[i] = 0
            elif self.board_curr[1, i] == 1:
                self.board_blots[i] = -1
                self.board_blocks[i] = 0
            elif self.board_curr[0, i] > 1:
               self.board_blots[i] = 0
               self.board_blocks[i] = 1
            elif self.board_curr[1, i] > 1:
               self.board_blots[i] = 0
               self.board_blocks[i] = 1
            else:
                if self.board_curr[0, i ] != 0 or self.board_curr[1, i] != 0:
                    raise ValueError("Unexpected values in update_state")
        
        self.bear_off[self.player] = self.can_bear_off()

    
    cpdef void set_player(self, int player):
        if self.player == player:
            return

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
            
        return MoveGenerator.generate_moves(self.board_curr, self.dice[0], self.dice[1])

    cpdef list get_legal_moves2(self):
        if self.player == NONE:
            raise ValueError("Error player is not set yet")

        if not self.is_dice_valid():
            raise ValueError(f"Error Dice not valid: dice: {self.dice}")
        
        if DEBUG:
            # Print dice in decimal format
            print(f"Player: {self.player} Valid Dice: {self.is_dice_valid()} Dice: [{self.dice[0]}, {self.dice[1]}]")
            
        return MoveGenerator.generate_moves2(self.board_curr, self.dice[0], self.dice[1])

    cpdef list get_all_moves_all_dice(self):
        cdef list allDiceMoves = []

        if self.player == NONE:
            raise ValueError("Error player is not set yet")

        for i in range(21):
            dice = self.indexToDice(i)
            allDiceMoves.append(MoveGenerator.generate_moves2(self.board_curr, dice[0], dice[1]))
        
        return allDiceMoves
    
    
    cpdef void do_moves(self, MoveSequence moveSeq):
        cdef int i

        if self.player == NONE:
            raise ValueError("Player must be set before making moves")
        
        if moveSeq:
            self.move_seq_list.append(moveSeq.copy())
            #print(f"Applying Move Seq {self.player} MoveSeq: {moveSeq.dice[0]} {moveSeq.dice[1]}")
            for i in range(moveSeq.n_moves):
                
                BoardState.sanity_checks(self.board)
                if not BoardState.can_move_pip(self.board_curr, moveSeq.moves[i].src, moveSeq.moves[i].n):
                    #print(f"Invalid move: {moveSeq.moves[i].src} {moveSeq.moves[i].n}")
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

    cpdef tuple indexToDice(self, unsigned char index):
        cdef list diceLookup = [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6),
                            (2,2), (2,3), (2,4), (2,5), (2,6),
                            (3,3), (3,4), (3,5), (3,6),
                            (4,4), (4,5), (4,6), 
                            (5,5), (5,6),
                            (6,6)]
        return diceLookup[index]

    cpdef unsigned char diceToIndex(self, unsigned char d1, unsigned char d2):
        cdef unsigned char* d1_lookup = [0, 0, 6, 11, 15, 18, 20]
        
        if d1 > d2:
            d1, d2 = d2, d1  # Ensure d1 ≤ d2 for proper indexing

        cdef unsigned char d1_offset = d1_lookup[d1]  # Fix indexing issue
        cdef unsigned char d2_offset = d2 - d1

        return d1_offset + d2_offset




    
    

   