cimport numpy as np
from bg_common cimport *

cdef class BoardState:
    @staticmethod
    cdef bint can_bear_off(np.ndarray[np.int8_t, ndim=2] board)
    @staticmethod
    cdef bint can_move_pip(np.ndarray[np.int8_t, ndim=2] board, unsigned char src, unsigned char n)
    @staticmethod
    cdef bint apply_move(np.ndarray[np.int8_t, ndim=2] board, Move move)
    @staticmethod
    cdef void sanity_checks(np.ndarray[np.int8_t, ndim=2] board)
    @staticmethod
    cdef void undo_move(np.ndarray[np.int8_t, ndim=2] board, Move move, bint blotted)
    
