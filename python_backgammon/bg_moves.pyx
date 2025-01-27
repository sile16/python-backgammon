import numpy as np
cimport numpy as np
cimport cython
from bg_common cimport *
from bg_board cimport BoardState
from libc.stdint cimport int32_t, uint32_t

#import type list

# Bit-shifting constants
cdef int FROM_BITS = 5    # 32 positions (0-24 board + bar)
cdef int DICE_BITS = 3    # 8 possible dice values (1-6 + special cases)
cdef int MOVE_BITS = 8    # FROM_BITS + DICE_BITS

# Masks for bitwise operations
cdef uint32_t FROM_MASK = (1 << FROM_BITS) - 1
cdef uint32_t DICE_MASK = (1 << DICE_BITS) - 1
cdef uint32_t MOVE_MASK = (1 << MOVE_BITS) - 1

# Special constants
cdef uint32_t DOUBLE_FLAG = (1 << 30)  # Flag for double moves

cdef class MoveSequence:

    def __cinit__(self, dice=None):
        self.reset()
        if dice is not None:
            self.dice[0] = dice[0]
            self.dice[1] = dice[1]

    cdef void reset(self):
        self.n_moves = 0
        self.has_final_board = False
        self.dice[0] = 0
        self.dice[1] = 0
        self.n_used_moves = 0

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

    def __hash__(self):
        """Python-compatible hash function."""
        return self._toIndex()

    #def __toString__(self):
    #    output = "MoveSeq moves: "
    #    for m in min(range(self.n_moves), len(self.moves)):
    #        output += f"[{self.moves[m].src}, {self.moves[m].n}]"
    #    return output

    def toIndex(self):
        """Encodes a move sequence into a unique 32-bit integer index."""
        return self._toIndex()

    cdef uint32_t _toIndex(self):
        """Encodes a move sequence into a unique 32-bit integer index."""
        """Actually could change to 3 bytes to capture all possibilities"""
        if self.n_moves == 0 or self.n_moves > 4:
            return 0

        

        cdef uint32_t encoded = 0
        cdef uint32_t moveVale = 0
        cdef int i
        cdef Move move
        
        cdef uint32_t die, movesrc

        #print(f"toIndex: moves: {self.moves}")

        # Handle doubles
        if self.n_moves > 1 and self.moves[0].n == self.moves[1].n:
            #print("doubles")
            die = self.moves[0].n

            # Encode each move for doubles
        
            for i in range(4):
                if i < self.n_moves:
                    move = self.moves[i]
                    movesrc = move.src
                else:
                    movesrc = 25  # impossible source, this indicate no more moves
                
                # Shift existing bits and add new move
                encoded |= ( movesrc & FROM_MASK) << (i * FROM_BITS)
            
            # Add dice value and double marker
            
            encoded = (encoded << DICE_BITS) | die
            #print encoded in hex and binar
            #print(f"encoded: {hex(encoded)} , {bin(encoded)}")
            encoded |= DOUBLE_FLAG
            #print("add double flag")
            #print(f"encoded: {hex(encoded)} , {bin(encoded)}")
            
            return encoded

        # Handle regular moves
        #print("not doubles")
        
        for i in range(self.n_moves): 
            # Validate move
            move = self.moves[i]
            if move.src > 24 or move.n > 6 :
                return 0
            
            die = move.n
            movesrc = move.src

            
            moveVal = (movesrc << DICE_BITS) | die
            
            # Shift and add to encoded value
            encoded |= moveVal << (i * MOVE_BITS)
        
        return encoded

    @staticmethod
    def toSequenceFromIndex(encoded):
        """Decodes a uint32_t back into a move sequence."""
        return MoveSequence._toSequenceFromIndex(encoded)
    
    @staticmethod
    cdef MoveSequence _toSequenceFromIndex(uint32_t encoded):
        """Decodes a uint32_t back into a move sequence."""
        
        # Create an empty move sequence
        cdef MoveSequence sequence = MoveSequence()
        cdef int i = 0
        cdef uint32_t dice, movesCount, moveVal, from_pos, dice_val
        cdef Move move
        
        # Check if it's a double move, print the hex valu and binary
        #print(f"checking for doubles {hex(encoded)} , {bin(encoded)}, {bin(DOUBLE_FLAG)}")
        if encoded & DOUBLE_FLAG != 0:
            #print("doubles")
            # Remove the double offset flag
            encoded &= ~DOUBLE_FLAG  # Clear the double flag

            # Extract dice value
            dice = encoded & DICE_MASK
            #print(f"dice: {dice}")
            if dice == 0:
                return sequence  # Invalid dice value

            encoded >>= DICE_BITS  # Shift right to remove dice

            # Extract moves
            
            for i in range(4):
                from_pos = encoded & FROM_MASK
                #print(f"from_pos: {from_pos}")
                encoded >>= FROM_BITS
                
                if from_pos == 25:
                    break
                    
                sequence.add_move(from_pos, dice)

                if i < len(sequence.dice):
                    sequence.dice[i] = dice

            return sequence
        #print("Not doubles")

        # Handle regular moves
        movesCount = 0
        while encoded > 0 and movesCount < 4:
            moveVal = encoded & MOVE_MASK
            from_pos = (moveVal >> DICE_BITS) & FROM_MASK
            dice_val = moveVal & DICE_MASK

            #move = Move()
            #move.src = from_pos
            #move.n = dice_val
            sequence.add_move(from_pos, dice_val)
            if i < len(sequence.dice):
                sequence.dice[i] = dice_val

            encoded >>= MOVE_BITS
            movesCount += 1

        return sequence

    cpdef list get_moves_tuple(self):
        """Return a list of move tuples (src, n)"""
        cdef list result = []
        cdef int i
        for i in range(self.n_moves):
            result.append((self.moves[i].src, self.moves[i].n))
        return result

    #cdef unsigned char[4] get_moves(self):
    #    """Return a pointer to the moves array"""
    #    return self.moves
    
    cpdef MoveSequence add_move(self, unsigned char src, unsigned char n):
        if self.n_moves < 4:
            #print(f"Adding move: {src} {n}")
            self.moves[self.n_moves].src = src
            self.moves[self.n_moves].n = n
            self.n_moves += 1
        return self

    cpdef int use_move(self, unsigned char src, unsigned char n):
        """ removes move the first position """

        if self.n_moves == 0:
            raise ValueError("No moves to use")
        if self.moves[0].n != n or self.moves[0].src != src:
            raise ValueError("Move does not match first move in sequence")
        
        for i in range(0, self.n_moves - 1):
            self.moves[i] = self.moves[i + 1]
        
        self.n_moves -= 1

        self.used_moves[self.n_used_moves].src = src
        self.used_moves[self.n_used_moves].n = n
        self.n_used_moves += 1
        return self.n_moves

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

        for i in range(self.n_used_moves):
            new_seq.used_moves[i] = self.used_moves[i]
        
        if self.has_final_board:
            new_seq.set_final_board(self.final_board)
       
        return new_seq
    
    cdef void set_final_board(self, np.ndarray[np.int8_t, ndim=2] board):
        #if board != None:
        self.final_board = board.copy()
        self.has_final_board = True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MoveGenerator:
    """Static methods for generating legal moves"""
    @staticmethod
    cdef list generate_moves4(
        np.ndarray[np.int8_t, ndim=2] board,
        unsigned char d1, #D1 is always higher, enforced by set_dice
        unsigned char d2,
    ):
        """ new approach is to just find next dice useage not the full roll options"""
        """ we have to look ahead enough to find double dice usage vs single dice usage options"""
        cdef Move move
        cdef MoveSequence curr_seq
        cdef unsigned char move_num, src, dst, temp
        cdef bint found_valid_move, blotted
        cdef bint isDouble = d1 == d2
        cdef bint large_die_single = False

        cdef list single_moves = list()
        cdef list double_moves = list()

        cdef int idx = 0
        cdef int aidx

        # Process both dice orders if not a double
        if isDouble:
            idx = 0
            for x in range(BAR_POS, BEAR_OFF_POS):
                if board[0, x] > 0:
                    if BoardState.can_move_pip(board, x, d1):
                        single_moves.append(idx)
                        if d2 > 0:
                            #d2 is remaining moves
                            single_moves.append(idx + 15)
                    
                    idx += 1 # only increment idx on points with 1 or more pips, 
            
            #if DEBUG:
            #    for x in single_moves:
            #        assert x > 0 and x < 30

            if len(single_moves) == 0:
                single_moves.append(30)

            return single_moves 


        
        offset = 0
        offset2 = 15

        if d2 > d1:
            temp = d1
            d1 = d2
            d2 = temp
            offset = 15
            offset2 = 0

    
        max_moves = 0
        max_die = 0
        cdef Move m
        cdef np.ndarray orig
        
        a_idx = 0
        for a in range(BAR_POS, BEAR_OFF_POS):
            if board[0, a] == 0:
                continue
            
            elif board[0, a] > 0: #need to keep track of a_idx
                #print(f"a: {a}")
                #print(f"checking a can move {a} {d1}")
                if BoardState.can_move_pip(board, a, d1):
                    #print(f" Yes: a can move {a} {d1}")
                    
                    found_double = False
                    if d2 > 0:
                        m.src = a
                        m.n = d1
                        #orig = board.copy()
                        a_blotted = BoardState.apply_move(board, m)
                        for b in range(BAR_POS, BEAR_OFF_POS):
                            if BoardState.can_move_pip(board, b, d2):
                                found_double = True
                                break 

                        BoardState.undo_move(board, m, a_blotted)
                        #if not np.array_equal(orig, board):
                        #    raise ValueError("Board not returned to original state")

                    if found_double:
                        double_moves.append(a_idx + offset)
                        
                        #print(f"1Found double move: pos {a} a_idx {a_idx} offset {offset}  dice {dice_orders[0]}")
                    elif len(double_moves) == 0: # don't need to check max die, because we are highest
                        # it is a single, but if we've already found a doulbe we don't need to track
                        # that's why we check len(double_moves) == 0
                        #print(f"2Found single move: pos {a} a_idx {a_idx} offset {offset}  dice {dice_orders[0]}")
                        single_moves.append(a_idx + offset)

                #increment a_idx only if we have a pip on the point
                a_idx += 1

        a_idx = 0

        if len(single_moves) > 0:
            large_die_single_found = True
    
            
        if d2 > 0:
            for a in range(BAR_POS, BEAR_OFF_POS):

                if board[0, a] == 0:
                    continue
                elif board[0, a] > 0: # need this to keep track of a_idx

                    #print(f"7checking a can move {a} {d2}")
                    if BoardState.can_move_pip(board, a, d2):
                        found_double = False
                        if d1 > 0:
                            #check for second dice
                            m.src = a
                            m.n = d2
                            #orig = board.copy()
                            a_blotted = BoardState.apply_move(board, m)
                            
                            for b in range(BAR_POS, BEAR_OFF_POS):
                                if board[0, b] > 0 :
                                    #print(f"8checking b can move {b} {d1}")
                                    if BoardState.can_move_pip(board, b, d1):
                                        found_double = True
                                        #print("9found double")
                                        break
                            BoardState.undo_move(board, m, a_blotted)
                            #if not np.array_equal(orig, board):
                            #    raise ValueError("Board not returned to original state")

                        if found_double:
                            double_moves.append(a_idx + offset2)
                            #print(f"3Found double move: pos {a} a_idx {a_idx} offset {offset2}  dice {d2}")
                        elif len(double_moves) == 0 and not large_die_single_found:
                            # so no double found, so only add to list if we are max die
                            single_moves.append(a_idx + offset2)
                            #print(f"4Found single move: pos {a} a_idx {a_idx} offset {offset2}  dice {d2}")

                    #increment a_idx only if we have a pip on the point    
                    a_idx += 1
            
        if len(double_moves) > 0:
            #print(f"double moves: {double_moves}")
            return double_moves
        elif len(single_moves) > 0:
            #print("single moves: ", single_moves)
            return single_moves
        else:
            #return a pass move, no moves found
            single_moves.append(30)
            return single_moves

        #filter is uneeded as we are only looking for the next move
        
    ## everything below here is old
    ## Old and not used but keep for reference and test case checking
    ## Todo: Remove later

    @staticmethod
    cdef list generate_moves(np.ndarray[np.int8_t, ndim=2] board, unsigned char d1, unsigned char d2):
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
        if len(all_sequences) == 0:
            all_sequences.append(MoveSequence())
            #print(f"going to return empty sequence: {len(all_sequences)}")
            return all_sequences

        return MoveGenerator._filter_moves(all_sequences)
        

    
    @staticmethod
    cdef void _generate_moves_recursive(
        np.ndarray[np.int8_t, ndim=2] board,
        unsigned char move_num,
        unsigned char d1,
        unsigned char d2,
        MoveSequence curr_sequence,
        list all_sequences
    ):
        cdef int src
        cdef Move move
        cdef np.ndarray[np.int8_t, ndim=2] new_board
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
        #print(f"Prefilter_1 list length: {len(sequences)} max_moves: {max_moves} max_die: {max_die}")
        
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
            
            board_hash = seq.final_board.tobytes()
            
            if board_hash not in unique_states:
                unique_states.add(board_hash)
                unique_sequences.append(seq)
        
        #print(f"Returning from filter_moves {len(unique_sequences)}, total all_sequences: ")
        return unique_sequences


    @staticmethod
    cdef list generate_moves2(np.ndarray[np.int8_t, ndim=2] board, unsigned char d1, unsigned char d2):
        
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
        if len(all_sequences) == 0:
            all_sequences.append(MoveSequence())
            return all_sequences

        return MoveGenerator._filter_moves2(all_sequences, max_moves, max_die, True)

    @staticmethod
    cdef list generate_moves3(np.ndarray[np.int8_t, ndim=2] board, unsigned char d1, unsigned char d2):
        
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
        
        return MoveGenerator._filter_moves2(all_sequences, max_moves, max_die, False)

    @staticmethod
    cdef void _generate_moves_iterative(
        np.ndarray[np.int8_t, ndim=2] board,
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
            stack.append((board, MoveSequence([d2, d1]), 0, BAR_POS))
            stack.append((board, MoveSequence([d1, d2]), 0, BAR_POS))
            
        elif d2 > d1:
            stack.append((board, MoveSequence([d1, d2]), 0, BAR_POS))
            stack.append((board, MoveSequence([d2, d1]), 0, BAR_POS))
            
        else:
            #they are the same
            stack.append((board, MoveSequence([d1, d2]), 0, BAR_POS))
        
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
                        stack.append((new_board, new_seq, move_num + 1, src))
            
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
                    all_sequences.append(curr_seq)
    
    
    @staticmethod
    cdef list _filter_moves2(list sequences, unsigned char max_moves, unsigned char max_die, bint filter):

        
        #print(f"Prefilter list length: {len(sequences)} max_moves: {max_moves} max_die: {max_die}")
        
        if not sequences:
            return []
            
        cdef set unique_states = set()
        cdef list filtered_sequences = []
        cdef bytes board_hash
        cdef MoveSequence seq

        if max_moves == 1:
            # checks for max die requirement
            for seq in sequences:
                if seq.moves[0].n == max_die:
                    filtered_sequences.append(seq)

            return filtered_sequences
        
        
        for seq in sequences:
            # this filters only moves which are max_moves
            if seq.n_moves == max_moves:
                if not seq.has_final_board:
                    raise ValueError("Sequence missing final board state")
                
                if filter:
                    board_hash = seq.final_board.tobytes()
                    if  board_hash not in unique_states:
                        unique_states.add(board_hash)
                        filtered_sequences.append(seq)
                else:
                    # even if the end board in the same we want to keep all possible move seq as they could be different
                    # for when doing indidivual pip moving
                    filtered_sequences.append(seq)
        
        #print(f"filtered list length: {len(filtered_sequences)} max_moves: {max_moves} max_die: {max_die}")
        return filtered_sequences
