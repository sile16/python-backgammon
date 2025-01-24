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
__all__ = ['set_debug', 'get_debug', 'State']

# Add global function to set debug mode
def set_debug(enabled):
    """Python-accessible wrapper for debug mode setting"""
    global DEBUG
    DEBUG = enabled

def get_debug():
    """Python-accessible wrapper for getting debug mode"""
    global DEBUG
    return DEBUG


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DiceRollHelper:
    """Helper class for DiceRoll operations"""
    
    @staticmethod
    cdef void create(unsigned char[2] dice, unsigned char d1, unsigned char d2) nogil:
        dice[0] = d1
        dice[1] = d2
    
    @staticmethod
    cdef tuple random() :
        return ( 1 + (rand() % 6), 1 + (rand() % 6) )

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class BGGame:
    """Main game state class"""
    cdef:
        public np.ndarray board
        public np.ndarray board_curr
        public np.ndarray blots
        public np.ndarray blocks

        public bint[2] bear_off
        public int player
        public int winner
        public tuple dice
        public np.ndarray remaining_dice
        public int n_legal_remaining_dice # could be less than reamining based on 
        public bint last_move
        public bint first_move
        public list legal_moves
        public int points
        public list move_seq_list
        public MoveSequence move_seq_curr
        
    
    def __cinit__(self):
        self.reset()

    def randomize_seed(self, seed = None):
        if seed:
            srand(seed)
        else:
            srand(int.from_bytes(os.urandom(4), 'little'))  # 32-bit random seed

    cpdef BGGame copy(self):
        cdef BGGame new_game = BGGame()
        
        new_game.set_board(self.board)
        new_game.player = self.player
        new_game.winner = self.winner
        new_game.dice = self.dice
        #new_game.dice[1] = self.dice[1]
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

        new_game.first_move = self.first_move
        new_game.last_move = self.last_move 
        return new_game

    cpdef list legal_actions(self):
        return MoveGenerator.generate_moves4(self.board_curr, self.remaining_dice[0], self.remaining_dice[1])

    cdef void _no_moves_left(self, MoveSequence mseq = None):
        self._check_for_winner()
        if not self.isTerminal():
            self.set_player(1 - self.player)
            self.roll_dice()
        self.legal_moves = []

    cpdef action_to_string (self, int action):
        cdef Move m = self.uncompress(action)
        return f"Move: From: {m.src} with Dice: {m.n}  landing on {min(m.src + m.n, BEAR_OFF_POS)}"

    cdef Move uncompress(self, int action):
        cdef int offset = 0
        cdef int dice_using = 0
        cdef int action_idx = 0

        if action > 14:
            action -= 15
            dice_using = 1

        #uncompresses action index to a point index
        for i in range(25):  # returns 0-24
            if self.board_curr[0, i] > 0:
                if action_idx == action:
                    return Move(i, self.remaining_dice[dice_using])
                action_idx += 1
        raise ValueError(f"Invalid action: {action} using_dice: {self.remaining_dice}")

    cpdef tuple step(self, int action):
        """
        action 0-14 is move src index of possible currnet user pips dice0 moves
        action 15-29 is move src position 0-24 dice1 moves
        applies move and then changes turn and rolls dice if turn is complete
        """
        cdef Move m
        cdef int dice_using = 0
        cdef int action_idx = 0
        cdef MoveSequence mSeq

        if action == ACTION_PASS:
            self._no_moves_left(MoveSequence())
            self.update_state()
            return ( self.get_observation(), self.points, self.isTerminal() )
        
       
        m = self.uncompress(action)
        #uncompresses action index to a point index        
        if m.n == 0:
            print(f"Board state:\n{self.board_curr}")
            raise ValueError(f"Invalid action: {action} using_dice: {self.remaining_dice} ")
    
    
        #print(f"Step action: action:{action} pos:{m.src }  dice:{m.n}  remaining_dice: {self.remaining_dice} n_legal_moves {self.n_legal_remaining_dice}")

    
        #print("about to apply move")

        BoardState.apply_move(self.board_curr, m)
        self.move_seq_curr.add_move(m.src, m.n)
        
        
        # move dice remaining down
        for i in range(dice_using, len(self.remaining_dice) - 1):
            self.remaining_dice[i] = self.remaining_dice[i + 1]
            self.remaining_dice[i + 1] = 0
        self.n_legal_remaining_dice -= 1
        
        if self.remaining_dice[0] == 0 or self.n_legal_remaining_dice == 0:
            #print(f"no more moves left in step()")
            # this is probably redudnat and we can just call fitler_move_from_legal_actions
            # and it should do the same thing
            
            self._no_moves_left(self.move_seq_curr.copy())
            self.move_seq_curr.reset()

        self.update_state()
        
        return ( self.get_observation(), self.points, self.isTerminal() )

    
    cpdef np.ndarray get_observation(self):
        """Returns a NumPy array representing the full game state."""
        cdef np.ndarray board_state = np.zeros(28, dtype=np.float32)  # Ensure float32 for consistency
        cdef np.ndarray combined

        # Fill board representation
        cdef int i
        for i in range(24):  
            board_state[i] = (self.board_curr[0, i + 1] - self.board_curr[1, i + 1]) / 15.0  # Normalize checkers

        # Bar and bear-off positions
        board_state[24] = self.board_curr[0, BAR_POS] / 15.0  # Player's checkers on bar
        board_state[25] = self.board_curr[0, BEAR_OFF_POS] / 15.0  # Player's checkers borne off
        board_state[26] = -self.board_curr[1, OPP_BAR_POS] / 15.0  # Opponent's checkers on bar
        board_state[27] = -self.board_curr[1, OPP_BEAR_OFF_POS] / 15.0  # Opponent's checkers borne off

        # Additional flags (turn status, legal moves, etc.)
        cdef np.ndarray additional_flags = np.array([
            1.0 if self.first_move else 0.0,  # Float to match observation format
            1.0 if self.last_move else 0.0,  # Float to match observation format
            1.0 if self.bear_off[self.player] else 0.0,
            0.0 if self.bear_off[1 - self.player] else 1.0,
            self.n_legal_remaining_dice / 4.0,  # Normalize to range [0,1]
        ], dtype=np.float32)

        # Ensure dice values are NumPy arrays before division
        cdef np.ndarray dice_values = np.array(self.dice, dtype=np.float32) / 6.0
        cdef np.ndarray remaining_dice_values = np.array(self.remaining_dice, dtype=np.float32) / 6.0

        # Convert blots and blocks to float32 for consistency
        cdef np.ndarray blots_float = self.blots.astype(np.float32)
        cdef np.ndarray blocks_float = self.blocks.astype(np.float32)

        # Concatenate into a single observation array
        combined = np.concatenate([
            board_state, 
            additional_flags, 
            dice_values, 
            remaining_dice_values, 
            blots_float, 
            blocks_float
        ])

        return np.expand_dims(np.expand_dims(combined, axis=0), axis=0)  # Add batch and channel dimensions

    
    cpdef int get_move_count(self):
        return len(self.move_seq_list)
    
    cpdef void render(self):
        #print the board to the console
        print(f"Player: {self.player} Dice: {self.dice} Remaining Dice: {self.remaining_dice}")
        print(f"Blots: {self.blots}")
        print(f"Blocks: {self.blocks}")
        print(f"Bear Off: {self.bear_off}")
        print(f"Move Seq List: {self.move_seq_list}")
        print(f"{self.board_curr}")
        print(f"Points: {self.points}")
        print(f"Winner: {self.winner}")
        print(f"Legal Moves: {self.legal_moves()}")
        print(f"n_legal_remaining_dice: {self.n_legal_remaining_dice}")
        #print(f"Last Move: {self.last_move}")
    
    cpdef void reset(self):
        self.player = NONE
        self.winner = NONE
        self.legal_moves = []
        self.points = 0
        self.dice = (0, 0)
        self.remaining_dice = np.zeros(4, dtype=np.int8)
        self.n_legal_remaining_dice = 0

        self.bear_off[0] = False
        self.bear_off[1] = False
        self.move_seq_curr = MoveSequence()
        self.move_seq_list = []
        

        #init blots and blocks
        self.blots = np.zeros(24, dtype=np.int8)
        self.blocks = np.zeros(24, dtype=np.int8)

        # Initialize board with starting position
        self.board = np.zeros((2, 26), dtype=np.int8)
        # Set starting positions for both players
        self.set_board([
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]
        ])


    cpdef void set_board(self, board_array):
        if isinstance(board_array, np.ndarray):
            self.board = board_array.copy()
        else:
            self.board = np.array(board_array, dtype=np.int8)
        
        self.set_player(self.player)
        self.update_state()
    
    cpdef np.ndarray get_board(self):
        """return board from current perspective"""
        return self.get_board_curr()

    cpdef np.ndarray get_board_curr(self):
        """Return board from current user's perspective"""
        # make a copy because board_curr changes
        return np.array(self.board_curr, dtype=np.int8) 

    cpdef np.ndarray get_board_white_black(self):
        return self.board

    cpdef np.ndarray get_board_black_white(self):
        return np.array(self.board[::-1, ::-1], dtype=np.int8)

    cpdef bint can_bear_off(self):
        return BoardState.can_bear_off(self.board_curr)

    cpdef void update_state(self):

        for i in range(24):
            #blots and blocks
            if self.board_curr[0, i + 1 ] == 1:
                self.blots[i] = -1
                self.blocks[i] = 0
            elif self.board_curr[1, i + 1] == 1:
                self.blots[i] = 1
                self.blocks[i] = 0
            elif self.board_curr[0, i + 1] > 1:
               self.blots[i] = 0
               self.blocks[i] = 1
            elif self.board_curr[1, i + 1] > 1:
               self.blots[i] = 0
               self.blocks[i] = -1
            else:
                if self.board_curr[0, i + 1] != 0 or self.board_curr[1, i + 1] != 0:
                    raise ValueError("Unexpected values in update_state")
        
        self.bear_off[self.player] = self.can_bear_off()

    
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
            self.dice = dice
            self.remaining_dice[0] = self.dice[0]
            self.remaining_dice[1] = self.dice[1]
            if self.dice[0] == self.dice[1]:
                self.remaining_dice[2] = self.dice[0]
                self.remaining_dice[3] = self.dice[1]
                self.n_legal_remaining_dice = 4
            else:
                self.remaining_dice[2] = 0
                self.remaining_dice[3] = 0
                self.n_legal_remaining_dice = 2
            self.first_move = True
            self.last_move = False
    
    cpdef void pick_first_player(self):
        while self.player == NONE:
            self.dice = DiceRollHelper.random()
            if not self.dice[0] == self.dice[1]:
                if self.dice[0] > self.dice[1]:
                    self.set_player(WHITE)
                else:
                    self.set_player(BLACK)
    
    cpdef void roll_dice(self):
        dice = DiceRollHelper.random()
        #print("Rolling Dice: ", dice)
        self.set_dice(dice)


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




    
    

   