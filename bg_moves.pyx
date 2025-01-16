import numpy as np
cimport numpy as np
cimport cython
from bg_common cimport *
from bg_board cimport BoardState

#import type list



cdef class MoveSequence:
    #cdef:
    #    public Move[4] moves
    #    public unsigned char[2] dice
    #    public unsigned char n_moves
    #    public np.ndarray final_board
    #    public bint has_final_board

    def __cinit__(self, dice=None):
        self.n_moves = 0
        self.has_final_board = False
        self.dice = [0, 0]
        if dice is not None:
            self.dice[0] = dice[0]
            self.dice[1] = dice[1]

    # implement compare function, so we can do movesequence1 == movesequence2
    # Implement compare function
    def __eq__(self, MoveSequence other):
        if not isinstance(other, MoveSequence):
            return False

        # Compare dice
        if self.dice[0] != other.dice[0] or self.dice[1] != other.dice[1]:
            return False

        # Compare number of moves
        if self.n_moves != other.n_moves:
            return False

        # Compare moves
        for i in range(self.n_moves):
            if self.moves[i].src != other.moves[i].src or self.moves[i].n != other.moves[i].n:
                return False

        # Compare final board
        if self.has_final_board != other.has_final_board:
            return False

        if self.has_final_board and not np.array_equal(self.final_board, other.final_board):
            return False

        return True




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
        if d1 > d2:
            curr_seq.dice[0] = d1
            curr_seq.dice[1] = d2
        else:
            curr_seq.dice[0] = d2
            curr_seq.dice[1] = d1
        
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

        #print(f"Returning from generate_moves, total all_sequences: {len(all_sequences)}")
        
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
        if len(all_sequences) > 10000:
            #print some debugging info:
            print(f"Warning all_sequences len limit reached {len(all_sequences)})")
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
            if DEBUG:
                print("Not double and move_number 2, adding to list and returning, existing _gen_move")
            return
        elif isDouble and move_num < 4:
            # For doubles, allow up to four moves
            die_value = curr_sequence.dice[0]
        elif move_num == 4:
            # End after four moves for doubles
            curr_sequence.set_final_board(board)
            all_sequences.append(curr_sequence.copy())
            if DEBUG:
                print("move_number 4, adding to list and returning, existing _gen_move")
            return
        else:
            if DEBUG:
                print(f"Invalid move_num {move_num} in recursive generation")
            return
        
        # Try all possible source positions
        for src in range(BAR_POS, BEAR_OFF_POS):
            if BoardState.can_move_pip(board, src, die_value):
                if DEBUG:
                    print(f"Found valid move: {src} {die_value}")

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
        if not found_valid_move and move_num > 0:  # have to have at least 1 valid move.
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
