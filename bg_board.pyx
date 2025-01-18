
import numpy as np
cimport numpy as np
from bg_common cimport *

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
        
        if src > 24:
            raise ValueError("Invalid src position for a move")
            
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

        if board[0, move.src] == 0:
            raise ValueError("Error: No piece to move")

        if board[0, BAR_POS] > 0 and move.src != BAR_POS:
            raise ValueError("Error: Must move from bar first")
        
        if move.src > 24:
            raise ValueError("Invalid src position for a move")
        
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

            if board[0,i] > 15 or board[1,i] > 15:
                print(f"Board state:\n{board}")
                print("[[0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5]]")
                print(np.array_str(board, precision=2, suppress_small=True))
                raise ValueError("Error: Invalid board state, piece count > 15")
        
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

        # WRONG: Only 1 player should have pieces on the bar
        # this is wrong, both players can have pieces on the, the move OFF a bar, could hit an opponent
        # if board[0, BAR_POS] > 0 and board[1, OPP_BAR_POS] > 0:
        #    print(f"Board state:\n{board}")
        #    print("[[0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5]]")
        #    print(np.array_str(board, precision=2, suppress_small=True))
        #    raise ValueError("Error: Invalid board state, both players have pieces on the bar")