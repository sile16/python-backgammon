import numpy as np
cimport numpy as np
cimport cython
from bg_common cimport *
from bg_board cimport BoardState

#import type list

cdef class MoveSequence:

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
        cdef MoveSequence new_seq = MoveSequence((self.dice[0], self.dice[1]))
        cdef int i
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
    cdef list generate_moves(np.ndarray[np.uint8_t, ndim=2] board, unsigned char d1, unsigned char d2):
        cdef list all_sequences = []
        cdef unsigned char[2] reverse_dice
        cdef MoveSequence curr_seq
    
        if DEBUG:
            print(f"Generating moves with dice {d1} {d2}")
        #print(f"Generating moves with dice {d1} {d2}")

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
        cdef unsigned char start_pos = BAR_POS

        move.src = 0
        move.n = 0
        
        if DEBUG:
            print(f"_gen_moves Move number: {move_num}, Dice: {curr_sequence.dice[0]} {curr_sequence.dice[1]}")

        # Limit recursion depth for safety
        if len(all_sequences) > 30000:
            #print some debugging info:
            raise ValueError(f"Warning all_sequences len limit reached {len(all_sequences)})")
            
            
        
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
            if move_num > 0:
                start_pos = curr_sequence.moves[move_num - 1].src
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

        #print("Total Sequence before filtering: ", len(sequences))
        
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


    @staticmethod
    cdef list generate_moves2(np.ndarray[np.uint8_t, ndim=2] board, unsigned char d1, unsigned char d2):
        
        cdef list all_sequences = []
        cdef unsigned char max_moves = 0
        cdef unsigned char max_die = 0
        
        if DEBUG:
            print(f"Generating moves with dice {d1} {d2}")
        #print(f"Generating moves with dice {d1} {d2}")

        # Create initial MoveSequence with dice ordered (higher die first)
        
        MoveGenerator._generate_moves_iterative(
            board,
            d1,
            d2,
            all_sequences,
            &max_moves,
            &max_die
        )
        #print(f"Returning from generate_moves2, total all_sequences: {len(all_sequences)} max_moves: {max_moves} max_die: {max_die}")
        return MoveGenerator._filter_moves2(all_sequences, max_moves, max_die)

    @staticmethod
    cdef void _generate_moves_iterative(
        np.ndarray[np.uint8_t, ndim=2] board,
        unsigned char d1,
        unsigned char d2,
        list all_sequences,
        unsigned char* max_moves_ptr,
        unsigned char* max_die_ptr
    ):
        cdef list stack = []
        cdef Move move
        cdef MoveSequence curr_seq, new_seq
        cdef unsigned char move_num, src, die_value, start_src
        cdef bint found_valid_move

        cdef bint isDouble = d1 == d2


        
        if d1 > d2:
            stack.append((board.copy(), MoveSequence([d2, d1]), 0, BAR_POS))
            stack.append((board.copy(), MoveSequence([d1, d2]), 0, BAR_POS))
            
        elif d2 > d1:
            stack.append((board.copy(), MoveSequence([d1, d2]), 0, BAR_POS))
            stack.append((board.copy(), MoveSequence([d2, d1]), 0, BAR_POS))
            
        else:
            #they are the same
            stack.append((board.copy(), MoveSequence([d1, d2]), 0, BAR_POS))
        
        while stack:
            curr_board, curr_seq, move_num, last_src = stack.pop()
            found_valid_move = False
            
            if isDouble:
                die_value = curr_seq.dice[0]
            else:
                die_value = curr_seq.dice[move_num]

            start_src = last_src if isDouble and move_num > 0 else BAR_POS
            
            for src in range(start_src, BEAR_OFF_POS):
                if BoardState.can_move_pip(curr_board, src, die_value):
                    found_valid_move = True
                    
                    if move_num + 1 > max_moves_ptr[0]:
                        max_moves_ptr[0] = move_num + 1
                        
                    
                    new_board = curr_board.copy()
                    move.src = src
                    move.n = die_value
                    BoardState.apply_move(new_board, move)
                    
                    new_seq = curr_seq.copy()
                    new_seq.add_move(move.src, move.n)
                    
                    #if new_seq.n_moves == 1 and die_value > max_die_ptr[0]:
                    #    max_die_ptr[0] = die_value
                    
                    #next_move_found = any(BoardState.can_move_pip(new_board, s, d1 if move_num + 1 == 0 else d2) for s in range(BAR_POS, BEAR_OFF_POS))
                    
                    if (not isDouble and move_num == 1) or (isDouble and move_num == 3) :#or not next_move_found:
                        if new_seq.n_moves >= max_moves_ptr[0]:
                            new_seq.set_final_board(new_board)
                            all_sequences.append(new_seq)
                    else:
                        #could be more moves to make put onto the stack
                        stack.append((new_board.copy(), new_seq, move_num + 1, src))
            
            if not found_valid_move and curr_seq.n_moves > 0:
                # add in curr sequence as it's now final as no further moves possible
                # this should catch when only 1 or 3 moves are possible
                if curr_seq.n_moves == 1:
                    if curr_seq.moves[0].n < max_die_ptr[0]:
                        continue # don't add this as we've already found a higher value sequence.
                    
                    max_die_ptr[0] = curr_seq.moves[0].n
                    #print("Max die found updated to: ", max_die_ptr[0])
                
                # filter out moves if they are less than max moves
                if curr_seq.n_moves >= max_moves_ptr[0]:
                    curr_seq.set_final_board(curr_board)
                    all_sequences.append(curr_seq.copy())

    @staticmethod
    cdef list _filter_moves2(list sequences, unsigned char max_moves, unsigned char max_die):
        
        #print(f"Prefilter list length: {len(sequences)} max_moves: {max_moves} max_die: {max_die}")
        
        if not sequences:
            return []
            
        cdef set unique_states = set()
        cdef list filtered_sequences = []
        cdef bytes board_hash
        cdef MoveSequence seq

        
        
        for seq in sequences:
            if seq.n_moves == max_moves:
                if not seq.has_final_board:
                    raise ValueError("Sequence missing final board state")
            
                if max_moves == 1 and seq.moves[0].n < max_die:
                    pass #continue
                
                board_hash = seq.final_board.tobytes()
                
                if board_hash not in unique_states:
                    unique_states.add(board_hash)
                    filtered_sequences.append(seq)

        # For single moves, prefer higher die values
        if max_moves == 1:
            
            filtered_sequences = []
            for seq in sequences:
                if seq.moves[0].n == max_die:
                    filtered_sequences.append(seq)

            sequences = filtered_sequences
        
        return filtered_sequences
